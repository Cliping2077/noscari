"""Persistent goals and learned progress estimates; no external execution tools."""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from math import sqrt
from pathlib import Path

from axiom.core.models import LearningReceipt, new_id
from axiom.frontier.models import Goal, MotivationConfig, Opportunity, validate_payload
from axiom.learning.policies import digest_state
from axiom.memory.store import EventStore, validate_event


@dataclass
class _State:
    config: MotivationConfig | None = None
    goals: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)
    outcomes: list = field(default_factory=list)
    completed: set = field(default_factory=set)
    learned_ids: list = field(default_factory=list)
    history: list = field(default_factory=list)
    spent: int = 0
    paused: bool = False
    allowed: bool = True

    def fitted(self):
        weights = {g.observation.topic: 0. for g in self.goals.values()}
        ids = []
        for event_id, sample in self.outcomes:
            if not sample.verified:
                continue
            topic = self.goals[self.attempts[sample.attempt_id]].observation.topic
            weights[topic] += self.config.alpha * (sample.progress-weights[topic])
            ids.append(event_id)
        return weights, ids

    def allowed_goal(self, goal_id):
        return (goal_id in self.goals and self.goals[goal_id].status == "active"
                and not self.paused and self.allowed and self.spent < self.config.budget)

    def apply(self, event):
        validate_event(event)
        if not event["active"] or not event["kind"].startswith("motivation_"):
            raise ValueError("motivation journal must contain active motivation events only")
        kind, event_id = event["kind"], event["id"]
        data = validate_payload(kind, event["payload"])
        if kind == "motivation_config":
            if self.config is not None:
                raise ValueError("configuration cannot change during replay")
            self.config = data
            return
        if self.config is None:
            raise ValueError("configuration must be the first event")
        if kind == "motivation_goal":
            duplicate = any(g.observation.topic == data.observation.topic and
                g.observation.subject_id == data.observation.subject_id for g in self.goals.values())
            if data.id in self.goals or duplicate or data.status != "active" or len(self.goals) >= 128:
                raise ValueError("invalid, duplicate or excess goal")
            self.goals[data.id] = data
            self.weights.setdefault(data.observation.topic, 0.)
        elif kind == "motivation_status":
            if data.goal_id not in self.goals:
                raise ValueError("unknown goal")
            self.goals[data.goal_id] = self.goals[data.goal_id].model_copy(update={"status":data.status})
        elif kind == "motivation_control":
            if data.paused is not None:
                self.paused = data.paused
            if data.allow_experiment is not None:
                self.allowed = data.allow_experiment
        elif kind == "motivation_attempt":
            if data.goal_id not in self.goals:
                raise ValueError("unknown goal in attempt")
            if not self.allowed_goal(data.goal_id):
                raise PermissionError("goal, permission, pause or budget prevents exploration")
            self.attempts[event_id] = data.goal_id
            self.spent += 1
        elif kind == "motivation_consumed":
            if self.spent + data.count > self.config.budget:
                raise ValueError("consumed budget exceeds quota")
            self.spent += data.count
        elif kind == "motivation_feedback":
            if data.attempt_id not in self.attempts or data.attempt_id in self.completed:
                raise ValueError("feedback needs a unique reserved attempt")
            self.outcomes.append((event_id, data))
            self.completed.add(data.attempt_id)
        elif kind == "motivation_learn":
            expected, ids = self.fitted()
            if (data.before != self.weights or list(data.previous_ids) != self.learned_ids
                or data.after != expected or list(data.learned_ids) != ids or ids == self.learned_ids):
                raise ValueError("training state does not match verified experience replay")
            self.history.append((self.weights.copy(), self.learned_ids.copy()))
            self.weights, self.learned_ids = expected, ids
        elif kind == "motivation_rollback":
            if not self.history:
                raise ValueError("no update to roll back")
            previous, self.learned_ids = self.history.pop()
            self.weights = {g.observation.topic: 0. for g in self.goals.values()} | previous


def _replay(events):
    state, seen = _State(), set()
    for event in events:
        validate_event(event)
        if event["id"] in seen:
            raise ValueError("duplicate event identity")
        seen.add(event["id"])
        state.apply(event)
    if state.config is None:
        raise ValueError("missing configuration")
    return state


class MotivationCore:
    """Single-writer experimental API. The trusted host verifies probe outcomes."""
    def __init__(self, store_path, *, budget=None):
        self.store = EventStore(store_path)
        try:
            events = self.store.events()
            if not events:
                config = MotivationConfig() if budget is None else MotivationConfig(budget=budget)
                self.store.append("motivation_config", config.model_dump(mode="json"))
                events = self.store.events()
            self._state = _replay(events)
            if budget is not None and (type(budget) is not int or self._state.config.budget != budget):
                raise ValueError("requested budget differs from persisted configuration")
        except Exception:
            self.store.close()
            raise

    @property
    def goals(self):
        return deepcopy(self._state.goals)

    @property
    def weights(self):
        return self._state.weights.copy()

    @property
    def spent(self):
        return self._state.spent

    @property
    def parameter_digest(self):
        return digest_state(self.weights)

    def _commit(self, kind, payload):
        event = dict(id=new_id(), kind=kind, payload=payload, active=True)
        future = deepcopy(self._state)
        future.apply(event)  # validate all transitions before writing
        self.store.append(kind, payload, event_id=event["id"])
        self._state = future
        return event["id"]

    def observe(self, topic, *, novelty=0., prediction_error=0., subject_id="synthetic", source="local-observer"):
        signal = Opportunity(topic=topic, novelty=novelty, prediction_error=prediction_error,
            subject_id=subject_id, source=source)
        for goal in self._state.goals.values():
            if goal.observation.topic == topic and goal.observation.subject_id == subject_id:
                return goal.id
        if signal.novelty < .5 and signal.prediction_error < .2:
            return None
        goal = Goal(id=new_id(), observation=signal, question=f"Can I improve my prediction of {topic}?")
        self._commit("motivation_goal", goal.model_dump(mode="json"))
        return goal.id

    def suggest(self):
        candidates = [dict(goal_id=g.id, topic=g.observation.topic, question=g.question,
                           channel="local_experiment", score=self.weights[g.observation.topic])
            for g in self._state.goals.values() if self._state.allowed_goal(g.id)]
        # Stable insertion-order tie breaking, independent of random UUID identities.
        return sorted(candidates, key=lambda row: -row["score"])

    def reserve(self, goal_id):
        """Reserve one local-experiment unit before the host executes it."""
        return self._commit("motivation_attempt", dict(goal_id=goal_id))

    def feedback(self, attempt_id, *, loss_before, loss_after, verified=False):
        return self._commit("motivation_feedback", dict(attempt_id=attempt_id,
            loss_before=loss_before, loss_after=loss_after, verified=verified))

    def learn(self):
        before, old_digest = self.weights, self.parameter_digest
        after, ids = self._state.fitted()
        if ids != self._state.learned_ids:
            self._commit("motivation_learn", dict(before=before, after=after,
                previous_ids=self._state.learned_ids, learned_ids=ids))
        delta = sqrt(sum((self.weights[k]-before[k])**2 for k in before))
        return LearningReceipt(updated=delta > 0, before_digest=old_digest,
            after_digest=self.parameter_digest, parameter_delta=delta, eligible_records=len(ids),
            reason="EMA of verified prediction progress; objective and question template are fixed")

    def set_status(self, goal_id, status):
        return self._commit("motivation_status", dict(goal_id=goal_id, status=status))

    def control(self, *, paused=None, allow_experiment=None):
        return self._commit("motivation_control", dict(paused=paused, allow_experiment=allow_experiment))

    def rollback(self):
        self._commit("motivation_rollback", {})

    def forget(self, subject_id):
        """Hard-delete local subject provenance; reset all learned derivatives.

        Keep anonymous consumption tombstones so deletion cannot refund execution.
        Previously exported files remain the caller's responsibility.
        """
        removed = {i for i,g in self._state.goals.items() if g.observation.subject_id == subject_id}
        attempts = {i for i,g in self._state.attempts.items() if g in removed}
        retained = []
        for event in self.store.events():
            kind, data = event["kind"], event["payload"]
            if kind in {"motivation_learn", "motivation_rollback"}:
                continue
            if kind == "motivation_attempt" and event["id"] in attempts:
                retained.append(dict(id=new_id(), kind="motivation_consumed", payload={"count":1}, active=True))
                continue
            if ((kind == "motivation_goal" and data["id"] in removed)
                or (kind == "motivation_status" and data["goal_id"] in removed)
                or (kind == "motivation_feedback" and data["attempt_id"] in attempts)):
                continue
            retained.append(event)
        future = _replay(retained)
        with self.store.db:
            self.store.db.execute("DELETE FROM events")
            self.store.db.executemany("INSERT INTO events(id,kind,payload,active) VALUES (?,?,?,?)",
                [(e["id"],e["kind"],json.dumps(e["payload"],allow_nan=False),1) for e in retained])
        self.store.db.execute("VACUUM")
        self._state = future

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(dict(schema_version="frontier-motivation-1",
            events=self.store.events()),ensure_ascii=False,allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, store_path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(data) != {"schema_version", "events"} or data["schema_version"] != "frontier-motivation-1":
            raise ValueError("unsupported motivation checkpoint")
        _replay(data["events"])
        store = EventStore(store_path)
        try:
            store.import_events(data["events"])
        finally:
            store.close()
        return cls(store_path)

    def close(self):
        self.store.close()
