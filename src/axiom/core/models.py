from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

Method = Literal["traverse", "semantic", "search", "ask"]
METHODS = ("traverse", "semantic", "search", "ask")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MenuLabel(Record):
    id: str
    label: str


class Observation(Record):
    task_id: str
    goal: str
    menus: tuple[MenuLabel, ...] = Field(min_length=1, max_length=64)
    budget: int = Field(ge=1, le=10000)
    allow_search: bool = True
    allow_ask: bool = True
    search_cost: int = Field(default=2, ge=1)
    ask_cost: int = Field(default=4, ge=1)
    search_reliability: float = Field(default=0.8, ge=0, le=1)
    teacher_reliability: float = Field(default=0.9, ge=0, le=1)
    risk: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def unique_menus(self):
        if len({m.id for m in self.menus}) != len(self.menus):
            raise ValueError("menu IDs must be unique")
        return self


class Demonstration(Record):
    observation: Observation
    method: Literal["semantic"] = "semantic"
    rule: dict[str, str] = Field(default_factory=dict)
    source: str = "teacher"
    subject_id: str = "synthetic"
    source_confidence: float = Field(default=1, ge=0, le=1)
    verified: StrictBool = False
    reward: float = Field(ge=-1, le=1)
    trace: tuple[str, ...] = ()


class Feedback(Record):
    observation: Observation
    method: Method
    reward: float = Field(ge=-1, le=1)
    verified: StrictBool = False
    source: str = "host"
    subject_id: str = "synthetic"
    prediction: float | None = None
    success: bool | None = None
    operations: int = Field(default=0, ge=0)


class Skill(Record):
    id: str
    method: Method
    version: int = 1
    source_event: str | None = None
    source: str
    conditions: str
    rule: dict[str, str] = Field(default_factory=dict)


class StoredFeedback(Feedback):
    features: tuple[float, ...] = Field(min_length=10, max_length=10)
    skill_id: str = Field(min_length=1)


class Candidate(Record):
    id: str
    method: Method
    score: float
    status: Literal["pending", "inactive"] = "pending"
    reason: str


class LearningReceipt(Record):
    updated: bool
    before_digest: str
    after_digest: str
    parameter_delta: float
    eligible_records: int
    reason: str


class Outcome(Record):
    success: bool
    operations: int
    menu_opens: int
    teacher_calls: int
    search_calls: int
    reward: float
    semantic_attempt: bool
    wrong_transfer: bool
    fallback_used: bool
    trace: tuple[str, ...]
    api_calls: int = 0
    api_cost_usd: float = 0.0


def new_id() -> str:
    return uuid4().hex
