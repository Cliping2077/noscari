"""Strict records for the minimal, bounded Motivation Core."""
from typing import Literal

from pydantic import Field, StrictBool

from axiom.core.models import Record

Status = Literal["active", "paused", "abandoned", "solved"]


class MotivationConfig(Record):
    budget: int = Field(default=256, ge=0, le=10000, strict=True)
    alpha: float = Field(default=.25, gt=0, le=1)


class Opportunity(Record):
    topic: str = Field(min_length=1, max_length=120)
    novelty: float = Field(default=0, ge=0, le=1)
    prediction_error: float = Field(default=0, ge=0)
    subject_id: str = Field(default="synthetic", min_length=1)
    source: str = "local-observer"


class Goal(Record):
    id: str
    observation: Opportunity
    question: str
    status: Status = "active"


class GoalStatus(Record):
    goal_id: str
    status: Status


class Control(Record):
    paused: StrictBool | None = None
    allow_experiment: StrictBool | None = None


class Attempt(Record):
    goal_id: str


class ProgressFeedback(Record):
    attempt_id: str
    loss_before: float = Field(ge=0)
    loss_after: float = Field(ge=0)
    verified: StrictBool = False

    @property
    def progress(self):
        return max(-1., min(1., (self.loss_before-self.loss_after)/max(1., self.loss_before)))


class MotivationUpdate(Record):
    before: dict[str, float]
    after: dict[str, float]
    previous_ids: tuple[str, ...]
    learned_ids: tuple[str, ...]


class ConsumedBudget(Record):
    count: int = Field(ge=1, strict=True)


class Empty(Record):
    pass


SCHEMAS = {
    "motivation_config": MotivationConfig, "motivation_goal": Goal,
    "motivation_status": GoalStatus, "motivation_control": Control,
    "motivation_attempt": Attempt, "motivation_feedback": ProgressFeedback,
    "motivation_learn": MotivationUpdate, "motivation_rollback": Empty,
    "motivation_consumed": ConsumedBudget,
}


def validate_payload(kind, payload):
    if kind not in SCHEMAS:
        raise ValueError("unknown motivation event")
    return SCHEMAS[kind].model_validate(payload)
