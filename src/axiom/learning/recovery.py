"""One-pass policy improvement from logged post-prefix Monte Carlo returns."""
import numpy as np

from axiom.core.recovery_models import RecoveryState
from axiom.environments.interaction import allowed_actions
from axiom.learning.policies import (FEATURE_NAMES, LinearPolicy, LinUCBPolicy,
                                     MLPPolicy, features)

RECOVERY_FEATURES = FEATURE_NAMES + (
    "remaining_fraction", "unopened_fraction", "remaining_per_unopened",
    "semantic_failed", "opened_fraction", "search_used", "ask_used",
    "hint_failures", "remaining_after_search", "remaining_after_ask")


def recovery_features(state, rule, initial_context=False):
    if initial_context:
        state = RecoveryState(base=state.base)
    obs = state.base
    unopened = len(obs.menus) - len(state.opened)
    dynamic = [state.remaining / obs.budget, unopened / len(obs.menus),
        state.remaining / max(1, unopened), float(state.semantic_failed),
        len(state.opened) / len(obs.menus), float("search" in state.help_used),
        float("ask" in state.help_used), state.hint_failures / 2,
        (state.remaining - obs.search_cost - 1) / obs.budget,
        (state.remaining - obs.ask_cost - 1) / obs.budget]
    return np.concatenate((features(obs, rule), np.array(dynamic, dtype=np.float64)))


def episode_rows(episode, initial_context=False):
    """No unchosen-action labels; future cost excludes already-spent prefix cost."""
    rows = []
    total = 0.0
    penalty = (0.6 + 0.2 * episode.initial.base.risk) / episode.initial.base.budget
    for transition in reversed(episode.transitions):
        cost = transition.after.spent - transition.before.spent
        total += float(transition.after.success) - penalty * cost
        rows.append((recovery_features(transition.before, episode.rule, initial_context).tolist(),
                     transition.action, total))
    return list(reversed(rows))


class RecoveryLinear(LinearPolicy):
    feature_dim = len(RECOVERY_FEATURES)

    def state(self):
        state = super().state()
        state.update(kind="recovery-linear-v1", features=list(RECOVERY_FEATURES))
        return state

    def restore(self, state):
        if (state.get("kind") != "recovery-linear-v1" or
            state.get("features") != list(RECOVERY_FEATURES)):
            raise ValueError("incompatible recovery feature schema")
        super().restore(dict(kind="linear", ridge=state.get("ridge"), weights=state.get("weights")))


class RecoveryLinUCB(LinUCBPolicy):
    feature_dim = len(RECOVERY_FEATURES)


class RecoveryMLP(MLPPolicy):
    feature_dim = len(RECOVERY_FEATURES)


def select_action(policy, state, rule, *, initial_context=False, inquiry=True):
    available = allowed_actions(state, rule, inquiry=inquiry)
    if not available:
        raise RuntimeError("no available actions in terminal state")
    scores = policy.scores(recovery_features(state, rule, initial_context))
    return sorted(available, key=lambda action: (-scores[action], action))[0]
