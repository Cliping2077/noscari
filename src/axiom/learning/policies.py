import hashlib
import json

import numpy as np

from axiom.core.models import METHODS, Observation
from axiom.environments.menu import semantic_guess

FEATURE_NAMES = ("bias", "semantic_match", "menus_over_8", "budget_over_10",
                 "search_cost_ratio", "ask_cost_ratio", "search_reliability",
                 "teacher_reliability", "risk", "budget_per_menu")


def features(obs: Observation, rule: dict[str, str], association: bool = True) -> np.ndarray:
    return np.array([1, float(semantic_guess(obs, rule) is not None) if association else 0,
        len(obs.menus) / 8, obs.budget / 10, obs.search_cost / obs.budget,
        obs.ask_cost / obs.budget, obs.search_reliability, obs.teacher_reliability,
        obs.risk, obs.budget / len(obs.menus)], dtype=np.float64)


def digest_state(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


class LinearPolicy:
    """Per-method ridge reward regression, refit on the validated replay buffer.

    argmin_W sum_i (x_i W[a_i] - r_i)^2 + ridge ||W - W0||^2.
    Priors encode only an initial traversal preference, not correct paths.
    """
    feature_dim = len(FEATURE_NAMES)

    def __init__(self, ridge: float = 1.0):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = ridge
        self.prior = np.zeros((len(METHODS), self.feature_dim))
        self.prior[:, 0] = [0.15, 0.0, -0.05, -0.05]
        self.weights = self.prior.copy()

    @property
    def digest(self):
        return digest_state(self.state())

    def scores(self, x):
        return dict(zip(METHODS, (self.weights @ x).tolist()))

    def fit(self, rows):
        self.weights = self.prior.copy()
        for index, method in enumerate(METHODS):
            samples = [(x, r) for x, a, r in rows if a == method]
            if not samples:
                continue
            X = np.array([x for x, _ in samples], dtype=np.float64)
            y = np.array([r for _, r in samples], dtype=np.float64)
            self.weights[index] = np.linalg.solve(
                X.T @ X + self.ridge * np.eye(self.feature_dim),
                X.T @ y + self.ridge * self.prior[index])

    def state(self):
        return {"kind": "linear", "ridge": self.ridge, "weights": self.weights.tolist()}

    def restore(self, state):
        if state.get("kind") != "linear" or state.get("ridge") != self.ridge:
            raise ValueError("incompatible policy state")
        weights = np.asarray(state["weights"], dtype=np.float64)
        if weights.shape != self.weights.shape or not np.isfinite(weights).all():
            raise ValueError("invalid policy weights")
        self.weights = weights.copy()


class LinUCBPolicy(LinearPolicy):
    """Disjoint-arm linear UCB; same logged feedback and feature representation."""
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.inverse = np.repeat(np.eye(self.feature_dim)[None, :, :], len(METHODS), axis=0)

    def fit(self, rows):
        super().fit(rows)
        for i, method in enumerate(METHODS):
            X = np.array([x for x, a, _ in rows if a == method], dtype=float)
            self.inverse[i] = np.linalg.inv(np.eye(self.feature_dim) + X.T @ X) if len(X) else np.eye(self.feature_dim)

    def scores(self, x):
        mean = self.weights @ x
        bonus = self.alpha * np.sqrt(np.maximum(0, np.einsum("i,aij,j->a", x, self.inverse, x)))
        return dict(zip(METHODS, (mean + bonus).tolist()))

    def state(self):
        return {"kind": "linucb", "weights": self.weights.tolist(),
                "inverse": self.inverse.tolist(), "alpha": self.alpha}


class MLPPolicy:
    """Optional CPU PyTorch comparator, 10->16->4, masked reward regression."""
    feature_dim = len(FEATURE_NAMES)

    def __init__(self, seed=0):
        import torch
        self.torch = torch
        self.seed = seed
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = torch.nn.Sequential(torch.nn.Linear(self.feature_dim, 16),
                torch.nn.Tanh(), torch.nn.Linear(16, len(METHODS)))
        self.initial = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    @property
    def digest(self):
        return digest_state(self.state())

    def scores(self, x):
        with self.torch.no_grad():
            values = self.model(self.torch.tensor(x, dtype=self.torch.float32)).tolist()
        return dict(zip(METHODS, values))

    def fit(self, rows):
        torch = self.torch
        self.model.load_state_dict(self.initial)
        if not rows:
            return
        X = torch.tensor([x for x, _, _ in rows], dtype=torch.float32)
        a = torch.tensor([METHODS.index(a) for _, a, _ in rows], dtype=torch.long)
        y = torch.tensor([r for _, _, r in rows], dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02, weight_decay=0.01)
        for _ in range(150):
            prediction = self.model(X).gather(1, a[:, None]).squeeze(1)
            loss = torch.nn.functional.mse_loss(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def state(self):
        return {"kind": "mlp", "seed": self.seed,
                "weights": {k: v.detach().tolist() for k, v in self.model.state_dict().items()}}
