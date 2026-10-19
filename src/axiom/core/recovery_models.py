"""Visible interaction state. These schemas contain no private menu placement."""
from pydantic import Field, StrictBool, model_validator

from axiom.core.models import Method, Observation, Record


class RecoveryConfig(Record):
    rule: dict[str, str] = Field(default_factory=dict)
    frozen: StrictBool = False
    initial_context: StrictBool = False
    inquiry: StrictBool = True


class RecoveryState(Record):
    base: Observation
    spent: int = Field(default=0, ge=0)
    opened: tuple[str, ...] = ()
    help_used: tuple[str, ...] = ()
    semantic_failed: StrictBool = False
    hint_failures: int = Field(default=0, ge=0, le=2)
    success: StrictBool = False

    @model_validator(mode="after")
    def consistent(self):
        ids = {m.id for m in self.base.menus}
        if len(set(self.opened)) != len(self.opened) or not set(self.opened) <= ids:
            raise ValueError("opened menus must be unique visible IDs")
        if len(set(self.help_used)) != len(self.help_used) or not set(self.help_used) <= {"search", "ask"}:
            raise ValueError("invalid help history")
        if not len(self.opened) <= self.spent <= self.base.budget:
            raise ValueError("inconsistent operation budget")
        if (self.semantic_failed or self.success) and not self.opened:
            raise ValueError("outcome requires an observed menu")
        return self

    @property
    def remaining(self):
        return self.base.budget - self.spent

    @property
    def done(self):
        return self.success or self.remaining == 0 or len(self.opened) == len(self.base.menus)


class RecoveryTransition(Record):
    before: RecoveryState
    action: Method
    after: RecoveryState

    @model_validator(mode="after")
    def consistent(self):
        a, b = self.before, self.after
        if a.done or a.base != b.base or b.spent <= a.spent:
            raise ValueError("transition must advance a live unchanged task")
        if not set(a.opened) <= set(b.opened) or not set(a.help_used) <= set(b.help_used):
            raise ValueError("observed evidence cannot disappear")
        opened = len(b.opened) - len(a.opened)
        cost = b.spent - a.spent
        if b.success and opened == 0:
            raise ValueError("success requires observing a new menu")
        if self.action in {"traverse", "semantic"}:
            if cost != 1 or opened != 1 or a.help_used != b.help_used:
                raise ValueError("menu action must open one new menu")
        else:
            channel_cost = a.base.ask_cost if self.action == "ask" else a.base.search_cost
            allowed = a.base.allow_ask if self.action == "ask" else a.base.allow_search
            if (not allowed or self.action in a.help_used or a.remaining <= channel_cost
                or set(b.help_used) != set(a.help_used) | {self.action}
                or opened not in {0, 1} or cost != channel_cost + opened):
                raise ValueError("invalid or unauthorized help transition")
        if b.semantic_failed != (a.semantic_failed or (self.action == "semantic" and not b.success)):
            raise ValueError("semantic failure must follow observed execution")
        if b.hint_failures != a.hint_failures + int(self.action in {"ask", "search"} and not b.success):
            raise ValueError("hint failure count must follow help outcome")
        return self


class RecoveryEpisode(Record):
    initial: RecoveryState
    prefix: RecoveryTransition | None = None
    transitions: tuple[RecoveryTransition, ...]
    final: RecoveryState
    rule: dict[str, str] = Field(default_factory=dict)
    verified: StrictBool = False
    subject_id: str = "synthetic"
    source: str = "simulator"

    @model_validator(mode="after")
    def contiguous(self):
        if self.prefix is not None and self.prefix.after != self.initial:
            raise ValueError("prefix must end at the initial recovery state")
        current = self.initial
        for transition in self.transitions:
            if transition.before != current:
                raise ValueError("episode transitions must be contiguous")
            current = transition.after
        if self.final != current or not self.final.done:
            raise ValueError("feedback requires a completed contiguous episode")
        return self
