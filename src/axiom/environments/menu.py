"""Bounded toy menu simulator. Opening a root reveals its submenu items.

The semantic vocabulary is a fixed capability prior, NOT learned language.
Only the environment and evaluator may access private target placement.
"""
import hashlib
import random

from axiom.core.models import Demonstration, MenuLabel, Observation, Outcome

CATEGORIES = ("file", "edit", "view", "image", "audio", "tools")
ALIASES = {"file": "document", "edit": "modify", "view": "display",
           "image": "picture", "audio": "sound", "tools": "utilities"}
FUNCTIONS = {
    "train": ("adjust", "configure", "inspect", "export"),
    "test": ("compose", "transform", "archive", "preview"),
    "demo": ("resize",),
}


def semantic_guess(obs: Observation, rule: dict[str, str]) -> str | None:
    category = obs.goal.split(":", 1)[0]
    expected = rule.get(category, category)
    acceptable = {expected, ALIASES.get(expected, expected)}
    return next((m.id for m in obs.menus if m.label.lower() in acceptable), None)


class MenuEnvironment:
    def __init__(self, seed: int, regime: str = "semantic", split: str = "test",
                 budget: int | None = None, help_condition: str = "standard"):
        if regime not in {"semantic", "scrambled", "legacy", "renamed"}:
            raise ValueError("unknown regime")
        if split not in FUNCTIONS:
            raise ValueError("unknown split")
        if help_condition not in {"standard", "cheap", "restricted"}:
            raise ValueError("unknown help condition")
        # Namespace separation applies to layouts as well as function text.
        rng = random.Random(f"{split}:{seed}")
        category = rng.choice(CATEGORIES)
        goal = f"{category}:{rng.choice(FUNCTIONS[split])}-{rng.randrange(100000)}"
        labels = list(CATEGORIES) + [f"extra-{i}" for i in range(rng.randrange(3))]
        rng.shuffle(labels)
        menus = [MenuLabel(id=f"menu-{i}", label=s) for i, s in enumerate(labels)]
        semantic_id = next(m.id for m in menus if m.label == category)
        target = semantic_id
        if regime == "scrambled":
            target = rng.choice(menus).id
        elif regime == "legacy" and rng.random() < 0.65:
            target = rng.choice([m for m in menus if m.id != semantic_id]).id
        elif regime == "renamed":
            menus = [m.model_copy(update={"label": ALIASES.get(m.label, m.label)
                     if rng.random() < 0.65 else f"panel-{i}"})
                     for i, m in enumerate(menus)]
        help_args = {}
        if help_condition == "cheap":
            help_args = dict(search_cost=1, ask_cost=1, search_reliability=0.65,
                             teacher_reliability=0.95)
        elif help_condition == "restricted":
            help_args = dict(allow_ask=False, search_cost=4, ask_cost=6,
                             search_reliability=0.5, teacher_reliability=0.6)
        identity = hashlib.sha256(f"{split}:{seed}:{regime}:{help_condition}".encode()).hexdigest()[:20]
        self._obs = Observation(task_id=identity, goal=goal, menus=tuple(menus),
                                budget=budget if budget is not None else rng.randint(4, 10),
                                risk=rng.choice([0.0, 0.5, 1.0]), **help_args)
        self._target = target
        self._items = {m.id: [f"other-{i}-{j}" for j in range(3)] for i, m in enumerate(menus)}
        self._items[target].append(goal)
        self._hint_draws = {channel: (rng.random(), rng.choice([m.id for m in menus if m.id != target]))
                            for channel in ("ask", "search")}
        self.spent = 0
        self.success = False
        self.opened: set[str] = set()
        self.help_used: set[str] = set()
        self.trace: list[str] = []

    def observe(self) -> Observation:
        return self._obs.model_copy(deep=True)

    def _charge(self, amount: int):
        if self.success or self.spent + amount > self._obs.budget:
            raise RuntimeError("episode complete or insufficient budget")
        self.spent += amount

    def open_menu(self, menu_id: str) -> tuple[str, ...]:
        if menu_id not in self._items:
            raise ValueError("unknown menu")
        self._charge(1)
        self.opened.add(menu_id)
        self.success = self._obs.goal in self._items[menu_id]
        self.trace.append(f"open:{menu_id}:{'found' if self.success else 'absent'}")
        return tuple(self._items[menu_id])

    def request_hint(self, channel: str) -> str:
        if channel not in {"ask", "search"}:
            raise ValueError("unknown channel")
        allowed = self._obs.allow_ask if channel == "ask" else self._obs.allow_search
        if not allowed:
            raise PermissionError("help channel not authorized")
        if channel in self.help_used:
            raise RuntimeError("help channel already used")
        cost = self._obs.ask_cost if channel == "ask" else self._obs.search_cost
        self._charge(cost)
        self.help_used.add(channel)
        reliability = self._obs.teacher_reliability if channel == "ask" else self._obs.search_reliability
        draw, incorrect = self._hint_draws[channel]
        hint = self._target if draw < reliability else incorrect
        self.trace.append(f"{channel}:hint:{hint}")
        return hint


def execute_method(env: MenuEnvironment, method: str, rule: dict[str, str]) -> Outcome:
    if method not in {"traverse", "semantic", "search", "ask"}:
        raise ValueError("unknown method")
    obs = env.observe()
    first = None
    if method == "semantic":
        first = semantic_guess(obs, rule)
    elif method in {"search", "ask"}:
        cost = obs.ask_cost if method == "ask" else obs.search_cost
        allowed = obs.allow_ask if method == "ask" else obs.allow_search
        if not allowed:
            raise PermissionError("help channel not authorized")
        if cost < obs.budget:
            first = env.request_hint(method)
    semantic_attempt = method == "semantic" and first is not None
    wrong = False
    if first is not None and env.spent < obs.budget:
        env.open_menu(first)
        wrong = semantic_attempt and not env.success
    fallback = method != "traverse" and not env.success
    for menu in obs.menus:
        if env.success or env.spent >= obs.budget:
            break
        if menu.id not in env.opened:
            env.open_menu(menu.id)
    # All groups pay the same operation budget, including help. No GPU/API units.
    reward = float(env.success) - (0.6 + 0.2 * obs.risk) * env.spent / obs.budget
    return Outcome(success=env.success, operations=env.spent, menu_opens=len(env.opened),
                   teacher_calls=int("ask" in env.help_used), search_calls=int("search" in env.help_used),
                   reward=reward, semantic_attempt=semantic_attempt, wrong_transfer=wrong,
                   fallback_used=fallback, trace=tuple(env.trace))


def make_demo(subject_id: str = "synthetic", partial_error: bool = False):
    # One witnessed successful example, with a declarative general rule. A partial
    # error outside the demonstrated category cannot be detected from that example.
    rule = {"image": "file"} if partial_error else {}
    for seed in range(100):
        env = MenuEnvironment(seed, split="demo", budget=8)
        if env.observe().goal.startswith("file:"):
            result = execute_method(env, "semantic", rule)
            return env.observe(), Demonstration(observation=env.observe(), rule=rule,
                verified=True, reward=result.reward, trace=result.trace, subject_id=subject_id)
    raise RuntimeError("demo generation failed")
