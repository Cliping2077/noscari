"""Private toy classroom. Teacher requests must only contain observe() results."""
import hashlib
import random

from axiom.core.models import MenuLabel, Observation
from axiom.environments.menu import MenuEnvironment


class ClassroomMenu(MenuEnvironment):
    def __init__(self, seed, *, split="test", regime="semantic", budget=4, allow_ask=True):
        if split not in {"lesson","test"} or regime not in {"semantic","scrambled","legacy"}:
            raise ValueError("invalid classroom configuration")
        rng = random.Random(f"classroom-v1:{split}:{seed}:{regime}")
        labels = ["Documents","Canvas","Arrangement","Sound desk","Workspace","Export"]
        labels += [f"Panel {i}" for i in range(rng.randrange(3))]
        rng.shuffle(labels)
        menus = tuple(MenuLabel(id=f"root-{i}",label=label) for i,label in enumerate(labels))
        target = next(m.id for m in menus if m.label == "Sound desk")
        if regime == "scrambled":
            target = rng.choice(menus).id
        elif regime == "legacy" and rng.random() < .65:
            target = rng.choice([m for m in menus if m.id != target]).id
        verbs = ("reduce-hiss",) if split == "lesson" else ("normalize-level","remove-hum","balance-stereo","fade-out")
        goal = f"audio:{rng.choice(verbs)}-{rng.randrange(100000)}"
        identity = hashlib.sha256(f"classroom-v1:{split}:{seed}:{regime}".encode()).hexdigest()[:24]
        self._obs = Observation(task_id=identity,goal=goal,menus=menus,budget=budget,
            allow_search=False,allow_ask=allow_ask,ask_cost=1,teacher_reliability=.95)
        self._target = target
        self._items = {m.id:[f"unrelated-{m.id}-{j}" for j in range(3)] for m in menus}
        self._items[target].append(goal)
        self.spent, self.success = 0, False
        self.opened, self.help_used, self.trace = set(), set(), []

    def request_hint(self, channel):
        raise PermissionError("classroom has no search or simulated teacher; use the external bridge")

    def reserve_teacher(self):
        if not self._obs.allow_ask or "ask" in self.help_used or self.spent+self._obs.ask_cost >= self._obs.budget:
            raise PermissionError("teacher unavailable or insufficient budget")
        self._charge(self._obs.ask_cost)
        self.help_used.add("ask")
        self.trace.append("ask:external-request-reserved")

    def export_private_case(self):
        """Evaluator-only serialization. Never sent through the teacher bridge."""
        return dict(observation=self._obs.model_dump(mode="json"),target=self._target,items=self._items)

    @classmethod
    def from_private_case(cls, case):
        obj = cls.__new__(cls)
        obj._obs = Observation.model_validate(case["observation"])
        obj._target, obj._items = case["target"], case["items"]
        ids = {m.id for m in obj._obs.menus}
        if set(obj._items) != ids or obj._target not in ids:
            raise ValueError("invalid private fixture")
        if any((obj._obs.goal in items) != (menu == obj._target) for menu,items in obj._items.items()):
            raise ValueError("private fixture has inconsistent target evidence")
        obj.spent, obj.success = 0, False
        obj.opened, obj.help_used, obj.trace = set(), set(), []
        return obj
