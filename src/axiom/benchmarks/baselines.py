import numpy as np

from axiom.core.models import METHODS
from axiom.learning.policies import LinearPolicy, digest_state


def available_methods(obs, *, association=True, inquiry=True):
    methods = ["traverse"]
    if association:
        methods.append("semantic")
    if inquiry:
        if obs.allow_search and obs.search_cost < obs.budget:
            methods.append("search")
        if obs.allow_ask and obs.ask_cost < obs.budget:
            methods.append("ask")
    return methods


class SerialPolicy:
    """No cross-task retained demonstration, experience or updated weights."""
    def scores(self, x):
        return dict(traverse=1.0, semantic=-1.0, search=-1.0, ask=-1.0)

    def state(self):
        return {"kind": "serial"}

    @property
    def digest(self):
        return digest_state(self.state())


class FixedPolicy(SerialPolicy):
    """Multiple proposals with a fixed inspectable heuristic, never fitted."""
    def scores(self, x):
        return dict(traverse=0.15, semantic=0.65 if x[1] else -0.5,
                    search=0.6*x[6] - 0.8*x[4], ask=0.6*x[7] - 0.8*x[5])

    def state(self):
        return {"kind": "fixed", "version": 1}


class MemoryPolicy(SerialPolicy):
    """Fixed k-nearest experience retrieval per method; no fitted weights.

    This is a memory/nonparametric baseline, not a claim that retrieval is incapable
    of learning in the broader statistical sense. Same logged rows as AXIOM.
    """
    def __init__(self, rows, k=8):
        self.rows, self.k = rows, k
        self.prior = LinearPolicy()
        self.groups = {}
        for method in METHODS:
            samples = [(x, r) for x, a, r in rows if a == method]
            self.groups[method] = (np.array([x for x, _ in samples]), np.array([r for _, r in samples]))

    def scores(self, x):
        scores = self.prior.scores(x)
        for method, (X, y) in self.groups.items():
            if len(X):
                distances = np.sum((X - x) ** 2, axis=1)
                indexes = np.argsort(distances, kind="stable")[:self.k]
                weights = 1 / (0.1 + distances[indexes])
                scores[method] = float(np.average(y[indexes], weights=weights))
        return scores

    def state(self):
        return {"kind": "memory", "k": self.k, "rows_digest": digest_state({"rows": self.rows})}
