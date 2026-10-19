"""Single-step execution; no automatic traversal fallback."""
from axiom.core.recovery_models import RecoveryEpisode, RecoveryState, RecoveryTransition
from axiom.environments.menu import semantic_guess


def allowed_actions(state, rule, inquiry=True):
    if state.done:
        return []
    actions = ["traverse"]
    guess = semantic_guess(state.base, rule)
    if guess is not None and guess not in state.opened:
        actions.append("semantic")
    for channel in ("search", "ask"):
        allowed = state.base.allow_ask if channel == "ask" else state.base.allow_search
        cost = state.base.ask_cost if channel == "ask" else state.base.search_cost
        if inquiry and allowed and channel not in state.help_used and cost < state.remaining:
            actions.append(channel)
    return actions


class MenuSession:
    def __init__(self, environment):
        if environment.spent:
            raise ValueError("session requires a fresh environment")
        self.env = environment
        self._state = RecoveryState(base=environment.observe())
        self.initial = self._state
        self.prefix = None
        self.transitions = []
        self.rule = None

    def observe(self):
        return self._state.model_copy(deep=True)

    def step(self, action, rule):
        before = self.observe()
        if before.done:
            raise RuntimeError("episode complete")
        if action not in allowed_actions(before, rule):
            raise PermissionError("action unavailable, unauthorized or unaffordable")
        if self.rule is not None and self.rule != rule:
            raise ValueError("rule cannot change during an episode")
        self.rule = dict(rule)
        if action == "traverse":
            target = next(m.id for m in before.base.menus if m.id not in before.opened)
        elif action == "semantic":
            target = semantic_guess(before.base, rule)
        else:
            target = self.env.request_hint(action)
        if target not in before.opened:
            self.env.open_menu(target)
        self._state = RecoveryState(base=before.base, spent=self.env.spent,
            opened=tuple(sorted(self.env.opened)), help_used=tuple(sorted(self.env.help_used)),
            semantic_failed=before.semantic_failed or (action == "semantic" and not self.env.success),
            hint_failures=before.hint_failures + int(action in {"ask", "search"} and not self.env.success),
            success=self.env.success)
        transition = RecoveryTransition(before=before, action=action, after=self.observe())
        self.transitions.append(transition)
        return transition

    def semantic_prefix(self, rule):
        if self.transitions or self.prefix is not None or self._state.spent:
            raise RuntimeError("prefix must be the first operation")
        self.rule = dict(rule)
        if "semantic" in allowed_actions(self._state, rule):
            self.prefix = self.step("semantic", rule)
            self.transitions.clear()
        self.initial = self.observe()
        return self.initial

    def episode(self, *, verified=False, subject_id="synthetic", source="simulator"):
        return RecoveryEpisode(initial=self.initial, prefix=self.prefix,
            transitions=tuple(self.transitions), final=self.observe(), rule=self.rule or {},
            verified=verified, subject_id=subject_id, source=source)
