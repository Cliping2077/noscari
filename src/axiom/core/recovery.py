"""Experimental journal-backed learner for stepwise menu recovery."""
import json
from pathlib import Path

import numpy as np

from axiom.core.models import Candidate, LearningReceipt, new_id
from axiom.core.recovery_models import RecoveryConfig, RecoveryEpisode, RecoveryState
from axiom.environments.interaction import allowed_actions
from axiom.learning.recovery import RecoveryLinear, episode_rows, recovery_features
from axiom.memory.store import EventStore, validate_event


def validate_journal(events, config):
    if not events or events[0]["kind"] != "recovery_config" or not events[0]["active"]:
        raise ValueError("journal requires an initial configuration")
    if events[0]["payload"] != config:
        raise ValueError("checkpoint and journal configuration differ")
    for i, event in enumerate(events):
        validate_event(event)
        if not event["kind"].startswith("recovery_") or (i > 0 and event["kind"] == "recovery_config"):
            raise ValueError("mixed journal configuration")
        if event["kind"] == "recovery_episode" and event["payload"]["rule"] != config["rule"]:
            raise ValueError("episode rule differs from journal configuration")


class RecoveryLearner:
    def __init__(self, store_path, *, rule=None, frozen=None, initial_context=None, inquiry=None):
        self.store = EventStore(store_path)
        supplied = {k:v for k,v in dict(rule=rule, frozen=frozen,
            initial_context=initial_context, inquiry=inquiry).items() if v is not None}
        try:
            events = self.store.events()
            if events:
                if events[0]["kind"] != "recovery_config":
                    raise ValueError("journal missing recovery configuration")
                config = RecoveryConfig.model_validate(events[0]["payload"]).model_dump(mode="json")
                if any(config[k] != v or (k != "rule" and type(v) is not bool) for k,v in supplied.items()):
                    raise ValueError("requested configuration differs from persisted configuration")
                validate_journal(events, config)
            else:
                config = RecoveryConfig.model_validate(supplied).model_dump(mode="json")
                self.store.append("recovery_config", config)
        except Exception:
            self.store.close()
            raise
        self._config = RecoveryConfig.model_validate(config)
        self.policy = RecoveryLinear()
        self.observation = None
        self.history, self.learned_ids = [], []
        self._restore()

    def _restore(self):
        for event in self.store.events():
            if not event["kind"].startswith("recovery_"):
                raise ValueError("recovery learner requires its own journal")
            validate_event(event)
            if not event["active"]:
                continue
            data = event["payload"]
            kind = event["kind"]
            if kind == "recovery_observe":
                self.observation = RecoveryState.model_validate(data)
            elif kind in {"recovery_learn", "recovery_rollback"}:
                if kind == "recovery_learn":
                    self.history.append((data["before"], data["previous_ids"]))
                elif self.history:
                    self.history.pop()
                self.policy.restore(data["after"])
                self.learned_ids = data["learned_ids"]
            elif kind == "recovery_reset":
                self.policy = RecoveryLinear()
                self.learned_ids, self.history = [], []

    @property
    def rule(self):
        return dict(self._config.rule)

    @property
    def frozen(self):
        return self._config.frozen

    @property
    def initial_context(self):
        return self._config.initial_context

    @property
    def inquiry(self):
        return self._config.inquiry

    @property
    def parameter_digest(self):
        return self.policy.digest

    def observe(self, state):
        state = RecoveryState.model_validate(state)
        event_id = self.store.append("recovery_observe", state.model_dump(mode="json"))
        self.observation = state.model_copy(deep=True)
        return event_id

    def suggest(self):
        if self.observation is None:
            raise RuntimeError("observe before suggest")
        state = self.observation
        scores = self.policy.scores(recovery_features(state, self.rule, self.initial_context))
        available = allowed_actions(state, self.rule, self.inquiry)
        return sorted([Candidate(id=f"{state.base.task_id}:{state.spent}:{a}", method=a,
            score=scores[a], reason="estimated future utility from verified recovery episodes")
            for a in available], key=lambda c: (-c.score, c.method))

    def feedback(self, episode):
        episode = RecoveryEpisode.model_validate(episode)
        if episode.rule != self.rule:
            raise ValueError("episode belongs to a different rule version")
        return self.store.append("recovery_episode", episode.model_dump(mode="json"))

    def teach(self, episode):
        """Record a witnessed trajectory; verification remains the host's duty."""
        return self.feedback(episode)

    def replay_rows(self):
        rows, ids = [], []
        for event in self.store.events(active_only=True):
            if event["kind"] != "recovery_episode":
                continue
            episode = RecoveryEpisode.model_validate(event["payload"])
            if episode.verified and episode.rule == self.rule and episode.transitions:
                rows.extend(episode_rows(episode, self.initial_context))
                ids.append(event["id"])
        return rows, ids

    def learn(self):
        rows, ids = self.replay_rows()
        before, before_digest = self.policy.state(), self.parameter_digest
        old = self.policy.weights.copy()
        fitted = bool(rows) and ids != self.learned_ids and not self.frozen
        if fitted:
            self.policy.fit(rows)
            self.store.append("recovery_learn", dict(before=before, after=self.policy.state(),
                previous_ids=self.learned_ids, learned_ids=ids))
            self.history.append((before, self.learned_ids))
            self.learned_ids = ids
        delta = float(np.linalg.norm(self.policy.weights - old))
        return LearningReceipt(updated=delta > 0, before_digest=before_digest,
            after_digest=self.parameter_digest, parameter_delta=delta, eligible_records=len(rows),
            reason="return-to-go regression" if fitted else "no new verified trajectories or frozen")

    def rollback(self):
        if not self.history:
            raise RuntimeError("no recovery update to roll back")
        state, ids = self.history[-1]
        self.store.append("recovery_rollback", dict(after=state, learned_ids=ids))
        self.history.pop()
        self.policy.restore(state)
        self.learned_ids = ids

    def retract(self, event_id):
        with self.store.db:
            cursor = self.store.db.execute("UPDATE events SET active=0 WHERE id=? AND kind='recovery_episode'", (event_id,))
            if not cursor.rowcount:
                raise ValueError("no recovery episode with that ID")
            self.store.db.execute("INSERT INTO events(id,kind,payload) VALUES (?,?,?)",
                (new_id(), "recovery_reset", "{}"))
        self.policy = RecoveryLinear()
        self.learned_ids, self.history = [], []

    def forget(self, subject_id):
        self.store.forget(subject_id)
        self.store.append("recovery_reset", {})
        self.policy = RecoveryLinear()
        self.learned_ids, self.history = [], []
        self.observation = None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema_version="recovery-1", config=dict(rule=self.rule,
            frozen=self.frozen, initial_context=self.initial_context, inquiry=self.inquiry),
            events=self.store.events())
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, store_path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != "recovery-1":
            raise ValueError("unsupported recovery schema")
        config = data["config"]
        if (set(config) != {"rule", "frozen", "initial_context", "inquiry"}
            or any(type(config[k]) is not bool for k in ("frozen", "initial_context", "inquiry"))
            or not isinstance(config["rule"], dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in config["rule"].items())):
            raise ValueError("invalid recovery configuration")
        validate_journal(data["events"], config)
        store = EventStore(store_path)
        try:
            store.import_events(data["events"])
        finally:
            store.close()
        return cls(store_path, **config)

    def close(self):
        self.store.close()
