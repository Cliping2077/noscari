"""Local teacher messages are data; they convey neither verification nor authority."""
from typing import Literal

from pydantic import Field

from axiom.core.models import Record
from axiom.core.recovery_models import RecoveryState


class TeacherRequest(Record):
    request_id: str = Field(min_length=1)
    learner_name: Literal["Noscari Baby"] = "Noscari Baby"
    selected_action: Literal["ask"] = "ask"
    state: RecoveryState
    known_methods: tuple[str, ...]
    question: str
    decision_reason: str
    candidates: tuple[dict[str, float | str], ...]
    chooser_digest: str


class TeacherReply(Record):
    request_id: str = Field(min_length=1)
    rule: dict[str, str] = Field(min_length=1, max_length=1)
    explanation: str = Field(min_length=1, max_length=3000)

    def validate_for(self, request):
        if self.request_id != request.request_id:
            raise ValueError("reply belongs to a different request")
        category = request.state.base.goal.split(":",1)[0]
        labels = {m.label.lower() for m in request.state.base.menus}
        if set(self.rule) != {category} or next(iter(self.rule.values())) not in labels:
            raise ValueError("reply must map the current category to a visible label")


def question_for(state):
    attempted = len(state.opened)
    return (f"My current goal is {state.base.goal}. I have opened {attempted} menu(s) "
        f"without completing it, and have {state.remaining} operation units left. "
        "Search is unavailable. I do not yet have a learned semantic method for "
        "these labels. Could you teach me how to choose a relevant menu for this "
        "kind of task, so I can try it myself?")
