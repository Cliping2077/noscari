import json
from pathlib import Path

import numpy as np

from axiom.core.models import (Candidate, Demonstration, Feedback, LearningReceipt,
                               Observation, Skill)
from axiom.learning.policies import LinearPolicy, features
from axiom.memory.store import EventStore, validate_event


class Axiom:
    """Trusted host API. `verified` must come from a trusted outcome validator.

    It is not a field that an untrusted teacher/LLM may assign. This core neither
    runs arbitrary teacher code nor grants any external execution authority.
    """
    def __init__(self, store_path, *, frozen=False, association=True, inquiry=True):
        self.store = EventStore(store_path)
        self.frozen, self.association, self.inquiry = frozen, association, inquiry
        self.policy = LinearPolicy()
        self.observation = None
        self.workspace = []
        self.history = []
        self.learned_ids = []
        self._restore_events()

    def _base_skills(self):
        self.skills = {m: Skill(id=f"builtin:{m}:1", method=m, source="builtin",
            conditions="authorized and sufficient budget" if m != "traverse" else "any menu layout")
            for m in ("traverse", "search", "ask")}
        self.skill_versions = list(self.skills.values())

    def _register_demo(self, event):
        demo = Demonstration.model_validate(event["payload"])
        skill = Skill(id=event["id"], method="semantic", source=demo.source,
            source_event=event["id"], version=1 + sum(s.method == "semantic" for s in self.skill_versions),
            conditions="root labels match the declared semantic rule; fallback otherwise", rule=demo.rule)
        self.skills["semantic"] = skill
        self.skill_versions.append(skill)

    def _restore_events(self):
        self._base_skills()
        for event in self.store.events(active_only=True):
            validate_event(event)
            data = event["payload"]
            if event["kind"] == "teach":
                self._register_demo(event)
            elif event["kind"] == "observe":
                self.observation = Observation.model_validate(data)
            elif event["kind"] in {"learn", "rollback"}:
                if event["kind"] == "learn":
                    self.history.append((data["before"], data["previous_ids"]))
                elif self.history:
                    self.history.pop()
                self.policy.restore(data["after"])
                self.learned_ids = data["learned_ids"]
            elif event["kind"] == "reset":
                self.policy = LinearPolicy()
                self.history = []
                self.learned_ids = []

    @property
    def rule(self):
        return dict(self.skills["semantic"].rule) if "semantic" in self.skills else {}

    @property
    def parameter_digest(self):
        return self.policy.digest

    def observe(self, observation: Observation):
        observation = Observation.model_validate(observation)
        self.observation = observation.model_copy(deep=True)
        self.workspace = []
        return self.store.append("observe", observation.model_dump(mode="json"))

    def suggest(self):
        if self.observation is None:
            raise RuntimeError("observe before suggest")
        obs = self.observation
        scores = self.policy.scores(features(obs, self.rule, self.association))
        if "semantic" in self.skills:
            skill_id = self.skills["semantic"].id
            supported = skill_id in self.learned_ids or any(
                e["id"] in self.learned_ids and e["kind"] == "feedback"
                and e["payload"]["skill_id"] == skill_id
                for e in self.store.events(active_only=True))
            if not supported:
                scores["semantic"] = 0.0
        candidates = []
        for method in self.skills:
            if method == "semantic" and not self.association:
                continue
            if method in {"search", "ask"}:
                allowed = obs.allow_ask if method == "ask" else obs.allow_search
                cost = obs.ask_cost if method == "ask" else obs.search_cost
                if not self.inquiry or not allowed or cost >= obs.budget:
                    continue
            candidates.append(Candidate(id=f"{obs.task_id}:{method}", method=method,
                score=scores[method], reason="estimated return from validated replay"))
        self.workspace = sorted(candidates, key=lambda c: (-c.score, c.method))
        return list(self.workspace)

    def teach(self, demonstration: Demonstration):
        demo = Demonstration.model_validate(demonstration)
        event_id = self.store.append("teach", demo.model_dump(mode="json"))
        self._register_demo(dict(id=event_id, payload=demo.model_dump(mode="json")))
        return event_id

    def feedback(self, feedback: Feedback):
        data = Feedback.model_validate(feedback)
        if data.method not in self.skills:
            raise ValueError("feedback refers to an unavailable method")
        # Freeze the feature view at experience time, not after later teaching.
        payload = data.model_dump(mode="json")
        payload["features"] = features(data.observation, self.rule, self.association).tolist()
        payload["skill_id"] = self.skills[data.method].id
        return self.store.append("feedback", payload)

    def replay_rows(self):
        rows, ids = [], []
        for event in self.store.events(active_only=True):
            data = event["payload"]
            if event["kind"] not in {"teach", "feedback"} or not data["verified"]:
                continue
            if data["method"] == "semantic":
                current = self.skills.get("semantic")
                used_id = event["id"] if event["kind"] == "teach" else data["skill_id"]
                if current is None or used_id != current.id:
                    continue
            if event["kind"] == "teach":
                demo = Demonstration.model_validate(data)
                x = features(demo.observation, demo.rule, self.association).tolist()
            else:
                x = data["features"]
            rows.append((x, data["method"], data["reward"]))
            ids.append(event["id"])
        return rows, ids

    def learn(self):
        rows, ids = self.replay_rows()
        before_digest = self.parameter_digest
        before = self.policy.state()
        old_weights = self.policy.weights.copy()
        updated = bool(rows) and ids != self.learned_ids and not self.frozen
        if updated:
            previous_ids = list(self.learned_ids)
            self.policy.fit(rows)
            self.history.append((before, previous_ids))
            self.learned_ids = ids
            self.store.append("learn", dict(before=before, after=self.policy.state(),
                previous_ids=previous_ids, learned_ids=ids))
        delta = float(np.linalg.norm(self.policy.weights - old_weights))
        return LearningReceipt(updated=updated, before_digest=before_digest,
            after_digest=self.parameter_digest, parameter_delta=delta,
            eligible_records=len(rows), reason="replay refit" if updated else "frozen, no new verified data, or empty replay")

    def rollback(self):
        if not self.history:
            raise RuntimeError("no update to roll back")
        state, ids = self.history.pop()
        self.policy.restore(state)
        self.learned_ids = ids
        self.store.append("rollback", dict(after=state, learned_ids=ids))

    def _invalidate(self):
        self.policy = LinearPolicy()
        self.history = []
        self.learned_ids = []
        self.workspace = []
        self.store.append("reset", {})
        self._base_skills()
        for event in self.store.events(active_only=True):
            if event["kind"] == "teach":
                self._register_demo(event)

    def retract(self, event_id):
        self.store.retract(event_id)
        self._invalidate()

    def correct(self, event_id, replacement):
        if isinstance(replacement, Demonstration):
            kind = "teach"
            payload = Demonstration.model_validate(replacement).model_dump(mode="json")
        elif isinstance(replacement, Feedback):
            kind = "feedback"
            if replacement.method not in self.skills:
                raise ValueError("feedback refers to an unavailable method")
            payload = Feedback.model_validate(replacement).model_dump(mode="json")
            payload.update(features=features(replacement.observation, self.rule, self.association).tolist(),
                           skill_id=self.skills[replacement.method].id)
        else:
            raise ValueError("replacement must be a Demonstration or Feedback")
        replacement_id = self.store.replace_event(event_id, kind, payload)
        self.policy = LinearPolicy()
        self.history, self.learned_ids, self.workspace = [], [], []
        self._restore_events()
        return replacement_id

    def forget(self, subject_id):
        self.store.forget(subject_id)
        self.observation = None
        self._invalidate()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(schema_version=1, config=dict(frozen=self.frozen,
            association=self.association, inquiry=self.inquiry), events=self.store.events())
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, store_path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint schema")
        config = data.get("config", {})
        if not isinstance(config, dict) or not set(config) <= {"frozen", "association", "inquiry"} or any(type(v) is not bool for v in config.values()):
            raise ValueError("invalid checkpoint config")
        # Validate on a fresh destination; never replace a user's existing state.
        store = EventStore(store_path)
        try:
            store.import_events(data["events"])
        finally:
            store.close()
        return cls(store_path, **config)

    def close(self):
        self.store.close()
