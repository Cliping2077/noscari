"""Repeated menu tasks: evaluator-only truth and a strict public observation boundary.

The category matcher and primitive skills are supplied capabilities, not language
learned by the cognitive model. Neither regime nor teacher quality is observable.
"""
import hashlib
import random
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from axiom.core.models import MenuLabel, Record

ACTIONS = ('traverse', 'semantic', 'ask')
CATEGORIES = ('audio', 'image')
OBS_FEATURES = ('bias', 'audio', 'image', 'menus', 'budget', 'remaining',
    'unopened', 'semantic_available', 'semantic_rank', 'ask_available',
    'ask_cost', 'ask_used', 'opened')
TOKEN_FEATURES = ('audio', 'image', 'traverse', 'semantic', 'ask', 'success',
    'cost', 'responded', 'done', 'semantic_checked', 'semantic_hit')


class PublicState(Record):
    task_id: str
    goal: str
    menus: tuple[MenuLabel, ...] = Field(min_length=2, max_length=64)
    budget: int = Field(ge=1, le=100)
    ask_cost: int = Field(ge=1, le=100)
    allow_ask: StrictBool
    spent: int = Field(default=0, ge=0)
    opened: tuple[str, ...] = ()
    ask_used: StrictBool = False
    success: StrictBool = False

    @model_validator(mode='after')
    def consistent(self):
        ids = {m.id for m in self.menus}
        if len(ids) != len(self.menus) or len(set(self.opened)) != len(self.opened):
            raise ValueError('duplicate menu ID')
        if not set(self.opened) <= ids or not len(self.opened) <= self.spent <= self.budget:
            raise ValueError('invalid observations or spent budget')
        if self.category not in CATEGORIES or (self.success and not self.opened):
            raise ValueError('unsupported category or unobserved success')
        return self

    @property
    def category(self):
        return self.goal.split(':', 1)[0]

    @property
    def semantic_id(self):
        return next((m.id for m in self.menus if m.label == self.category), None)

    @property
    def remaining(self):
        return self.budget - self.spent

    @property
    def done(self):
        return self.success or self.remaining == 0 or len(self.opened) == len(self.menus)

    @property
    def available(self):
        if self.done:
            return ()
        actions = ['traverse']
        if self.semantic_id is not None and self.semantic_id not in self.opened:
            actions.append('semantic')
        if self.allow_ask and not self.ask_used and self.ask_cost < self.remaining:
            actions.append('ask')
        return tuple(actions)


class Evidence(Record):
    before: PublicState
    action: Literal['traverse', 'semantic', 'ask']
    after: PublicState
    responded: StrictBool = False
    opened_menu: str | None = None
    verified: StrictBool = False

    @model_validator(mode='after')
    def consistent(self):
        a, b = self.before, self.after
        immutable = ('task_id', 'goal', 'menus', 'budget', 'ask_cost', 'allow_ask')
        if any(getattr(a, k) != getattr(b, k) for k in immutable) or self.action not in a.available:
            raise ValueError('action needs a live unchanged task')
        added = set(b.opened) - set(a.opened)
        expected = {self.opened_menu} if self.opened_menu is not None else set()
        if not set(a.opened) <= set(b.opened) or added != expected:
            raise ValueError('evidence must describe exactly the new observed menu')
        if self.cost != (a.ask_cost if self.action == 'ask' else 0) + len(added):
            raise ValueError('incorrect action cost')
        if self.action != 'ask' and (len(added) != 1 or self.responded):
            raise ValueError('menu action must observe one menu')
        if self.action == 'semantic' and self.opened_menu != a.semantic_id:
            raise ValueError('semantic action must inspect the visible matching menu')
        if self.action == 'traverse' and self.opened_menu != next(m.id for m in a.menus if m.id not in a.opened):
            raise ValueError('traversal must inspect the next unopened menu')
        if b.ask_used != (a.ask_used or self.action == 'ask') or (b.success and not added):
            raise ValueError('invalid outcome')
        if self.action == 'ask' and not self.responded and added:
            raise ValueError('refusal cannot reveal a menu')
        return self

    @property
    def cost(self):
        return self.after.spent - self.before.spent

    @property
    def reward(self):
        return float(self.after.success) - .6 * self.cost / self.before.budget


def observation_vector(state):
    state = PublicState.model_validate(state)
    unopened = [m.id for m in state.menus if m.id not in state.opened]
    rank = unopened.index(state.semantic_id) if state.semantic_id in unopened else len(state.menus)
    return [1., *[float(state.category == c) for c in CATEGORIES], len(state.menus)/8,
        state.budget/6, state.remaining/6, len(unopened)/8,
        float('semantic' in state.available), rank/8, float('ask' in state.available),
        state.ask_cost/3, float(state.ask_used), len(state.opened)/8]


def feedback_vector(event):
    event = Evidence.model_validate(event)
    checked = event.opened_menu is not None and event.opened_menu == event.before.semantic_id
    return [*[float(event.before.category == c) for c in CATEGORIES],
        *[float(event.action == a) for a in ACTIONS], float(event.after.success),
        event.cost/6, float(event.responded), float(event.after.done), float(checked),
        float(checked and event.after.success)]


def make_session(namespace, seed, regime, *, tasks=16):
    """Private fixture factory. Learner gets only ContextMenu.observe()/step()."""
    if regime not in {'reliable', 'unreliable', 'partial', 'changing'} or tasks < 1:
        raise ValueError('unsupported regime or task count')
    world = random.Random(f'{namespace}:{seed}:world')
    teacher_accuracy = world.choice((.9, .4))
    answer_probability = world.choice((.9, .5))
    direction = world.choice((0, 1))
    cases = []
    for index in range(tasks):
        rng = random.Random(f'{namespace}:{seed}:task:{index}')
        category = rng.choice(CATEGORIES)
        reliable = regime == 'reliable'
        if regime == 'partial':
            reliable = CATEGORIES.index(category) == direction
        if regime == 'changing':
            reliable = (index < tasks//2) == bool(direction)
        labels = list(CATEGORIES) + ['file', 'edit', 'view', 'tools'] + [f'panel-{i}' for i in range(rng.randrange(3))]
        rng.shuffle(labels)
        menus = [dict(id=f'menu-{i}', label=label) for i, label in enumerate(labels)]
        semantic = next(m['id'] for m in menus if m['label'] == category)
        probability = .9 if reliable else .1
        target = semantic if rng.random() < probability else rng.choice([m['id'] for m in menus if m['id'] != semantic])
        identity = hashlib.sha256(f'{namespace}:{seed}:{regime}:{index}'.encode()).hexdigest()[:24]
        verb = 'practice' if namespace.startswith('train') else 'unseen'
        public = dict(task_id=identity, goal=f'{category}:{verb}-{rng.randrange(1000000)}',
            menus=menus, budget=rng.choice((4, 6)), ask_cost=rng.choice((1, 3)), allow_ask=rng.random() < .8)
        cases.append(dict(public=public, target=target, teacher_accuracy=teacher_accuracy,
            answer_probability=answer_probability, answer_draw=rng.random(), teacher_draw=rng.random(),
            wrong_hint=rng.choice([m['id'] for m in menus if m['id'] != target]),
            evaluator=dict(regime=regime, semantic_probability=probability, task_index=index)))
    return cases


class ContextMenu:
    """An extension of the bounded menu protocol; never handed to a policy."""
    def __init__(self, private_case):
        from copy import deepcopy
        self._case = deepcopy(private_case)
        self._state = PublicState.model_validate(private_case['public'])
        ids = {m.id for m in self._state.menus}
        if private_case['target'] not in ids or private_case['wrong_hint'] not in ids:
            raise ValueError('unknown private menu')
        for key in ('teacher_accuracy', 'answer_probability', 'answer_draw', 'teacher_draw'):
            if not 0 <= private_case[key] <= 1:
                raise ValueError('invalid simulator probability')

    def observe(self):
        return self._state.model_copy(deep=True)

    def step(self, action):
        before = self.observe()
        if action not in before.available:
            raise PermissionError('action unavailable, unauthorized or unaffordable')
        responded, cost = False, 0
        if action == 'traverse':
            target = next(m.id for m in before.menus if m.id not in before.opened)
        elif action == 'semantic':
            target = before.semantic_id
        else:
            cost = before.ask_cost
            responded = self._case['answer_draw'] < self._case['answer_probability']
            target = (self._case['target'] if self._case['teacher_draw'] < self._case['teacher_accuracy']
                else self._case['wrong_hint']) if responded else None
        opened_menu = target if target is not None and target not in before.opened else None
        cost += int(opened_menu is not None)
        data = before.model_dump()
        data.update(spent=before.spent+cost,
            opened=before.opened+((opened_menu,) if opened_menu is not None else ()),
            ask_used=before.ask_used or action == 'ask',
            success=opened_menu is not None and opened_menu == self._case['target'])
        self._state = PublicState.model_validate(data)
        return Evidence(before=before, action=action, after=self.observe(), responded=responded,
            opened_menu=opened_menu, verified=True)
