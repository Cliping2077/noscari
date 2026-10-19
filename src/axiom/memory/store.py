import json
from pathlib import Path
import sqlite3

from axiom.core.models import Demonstration, Observation, StoredFeedback, new_id
from axiom.learning.policies import LinearPolicy


def validate_payload(kind, payload):
    if not isinstance(kind, str) or not isinstance(payload, dict):
        raise ValueError("event payload must be an object")
    if kind.startswith("motivation_"):
        from axiom.frontier.models import validate_payload as validate_motivation
        validate_motivation(kind, payload)
        return
    if kind.startswith("recovery_"):
        from axiom.core.recovery_models import RecoveryConfig, RecoveryEpisode, RecoveryState
        from axiom.learning.recovery import RecoveryLinear
        if kind == "recovery_config":
            RecoveryConfig.model_validate(payload)
        elif kind == "recovery_episode":
            RecoveryEpisode.model_validate(payload)
        elif kind == "recovery_observe":
            RecoveryState.model_validate(payload)
        elif kind == "recovery_reset":
            if payload:
                raise ValueError("reset payload must be empty")
        elif kind in {"recovery_learn", "recovery_rollback"}:
            required = {"after", "learned_ids"}
            if kind == "recovery_learn":
                required |= {"before", "previous_ids"}
            if set(payload) != required:
                raise ValueError("invalid recovery learning event")
            RecoveryLinear().restore(payload["after"])
            if kind == "recovery_learn":
                RecoveryLinear().restore(payload["before"])
            for key in ("learned_ids", "previous_ids"):
                if key in payload and (not isinstance(payload[key], list) or not all(isinstance(v, str) for v in payload[key])):
                    raise ValueError("invalid recovery provenance")
        else:
            raise ValueError("unknown recovery event kind")
        return
    schemas = {"teach": Demonstration, "observe": Observation, "feedback": StoredFeedback}
    if kind in schemas:
        schemas[kind].model_validate(payload)
    elif kind in {"learn", "rollback"}:
        keys = {"after", "learned_ids"}
        if kind == "learn":
            keys |= {"before", "previous_ids"}
        if set(payload) != keys:
            raise ValueError("invalid learning event schema")
        LinearPolicy().restore(payload["after"])
        if kind == "learn":
            LinearPolicy().restore(payload["before"])
        for key in ("learned_ids", "previous_ids"):
            if key in payload and (not isinstance(payload[key], list) or
                                   not all(isinstance(v, str) for v in payload[key])):
                raise ValueError("invalid learning provenance")
    elif kind == "reset":
        if payload:
            raise ValueError("reset payload must be empty")
    else:
        raise ValueError("unknown event kind")


def validate_event(event):
    if not isinstance(event, dict) or set(event) != {"id", "kind", "payload", "active"}:
        raise ValueError("invalid event schema")
    if not isinstance(event["id"], str) or not event["id"] or type(event["active"]) is not bool:
        raise ValueError("invalid event identity or active flag")
    validate_payload(event["kind"], event["payload"])


class EventStore:
    """Single-writer SQLite journal; JSONL is a portable explicit snapshot."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)")
        self.db.commit()

    def append(self, kind, payload, event_id=None):
        validate_payload(kind, payload)
        event_id = event_id or new_id()
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        with self.db:
            self.db.execute("INSERT INTO events(id,kind,payload) VALUES (?,?,?)", (event_id, kind, data))
        return event_id

    def events(self, active_only=False):
        query = "SELECT id,kind,payload,active FROM events"
        if active_only:
            query += " WHERE active=1"
        return [dict(id=i, kind=k, payload=json.loads(p), active=bool(a))
                for i, k, p, a in self.db.execute(query + " ORDER BY seq")]

    def retract(self, event_id):
        with self.db:
            cursor = self.db.execute("UPDATE events SET active=0 WHERE id=? AND kind IN ('teach','feedback')", (event_id,))
            if not cursor.rowcount:
                raise ValueError("no teaching/feedback event with that ID")

    def replace_event(self, event_id, kind, payload):
        """Atomic correction: validation precedes retraction and reset."""
        validate_payload(kind, payload)
        replacement_id = new_id()
        with self.db:
            cursor = self.db.execute("UPDATE events SET active=0 WHERE id=? AND kind IN ('teach','feedback')", (event_id,))
            if not cursor.rowcount:
                raise ValueError("no teaching/feedback event with that ID")
            self.db.execute("INSERT INTO events(id,kind,payload) VALUES (?,?,?)",
                (replacement_id, kind, json.dumps(payload, ensure_ascii=False, allow_nan=False)))
            self.db.execute("INSERT INTO events(id,kind,payload) VALUES (?,?,?)", (new_id(), "reset", "{}"))
        return replacement_id

    def forget(self, subject_id):
        ids = [e["id"] for e in self.events() if e["payload"].get("subject_id") == subject_id
               or e["kind"] in {"learn", "rollback", "observe", "recovery_learn", "recovery_rollback", "recovery_observe"}]
        with self.db:
            self.db.executemany("DELETE FROM events WHERE id=?", [(i,) for i in ids])
        self.db.execute("VACUUM")

    def export_jsonl(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for event in self.events():
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(destination)

    def import_events(self, events):
        if self.events():
            raise ValueError("replay destination must be empty")
        for event in events:
            validate_event(event)
        with self.db:
            for event in events:
                self.db.execute("INSERT INTO events(id,kind,payload,active) VALUES (?,?,?,?)",
                    (event["id"], event["kind"], json.dumps(event["payload"], allow_nan=False), int(event["active"])))

    @classmethod
    def replay(cls, jsonl_path, store_path):
        events = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        store = cls(store_path)
        try:
            store.import_events(events)
        except Exception:
            store.close()
            raise
        return store

    def close(self):
        self.db.close()
