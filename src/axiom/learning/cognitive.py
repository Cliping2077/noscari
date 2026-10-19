"""Experimental causal feedback state. Torch is optional to the base package.

Episodic records, explicit statistics, fast neural activations and slow learned
weights are separate. No environment instance or private truth enters this API.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from axiom.environments.contextual import (ACTIONS, CATEGORIES, OBS_FEATURES,
    TOKEN_FEATURES, Evidence, PublicState, feedback_vector, observation_vector)
from axiom.learning.policies import digest_state

KINDS = ('gru', 'mlp', 'context_mlp', 'linear')
STATE_DIM = 24
STAT_DIM = 12


class HistoryStatistics:
    """Specified baselines: semantic success by category; help success/response.

    Only an actually opened matching menu counts as a semantic probe. No private
    target or latent reliability is used, including after failed tasks.
    """
    def __init__(self):
        self.ema = np.full(4, .5)
        self.total = np.zeros(4)
        self.count = np.zeros(4)

    def update(self, event):
        event = Evidence.model_validate(event)
        if not event.verified:
            raise ValueError('unverified evidence')
        updates = []
        if event.opened_menu is not None and event.opened_menu == event.before.semantic_id:
            updates.append((CATEGORIES.index(event.before.category), float(event.after.success)))
        if event.action == 'ask':
            updates.extend(((2, float(event.after.success)), (3, float(event.responded))))
        for index, value in updates:
            self.ema[index] += .25 * (value - self.ema[index])
            self.total[index] += value
            self.count[index] += 1

    def vector(self):
        return np.concatenate((self.ema, (self.total+1)/(self.count+2),
            self.count/(self.count+5))).tolist()

    def scores(self, obs):
        probability = float(self.ema[CATEGORIES.index(obs.category)])
        remaining = [m.id for m in obs.menus if m.id not in obs.opened]
        semantic_present = obs.semantic_id in remaining
        # A generic one-step probability/cost heuristic, never a regime switch.
        other = (1-probability)/max(1, len(remaining)-1) if semantic_present else 1/len(remaining)
        traversal = probability if remaining[0] == obs.semantic_id else other
        return dict(traverse=traversal-.6/obs.budget,
            semantic=probability-.6/obs.budget,
            ask=float(self.ema[2])-.6*(obs.ask_cost+1)/obs.budget)


def validated_sequences(sequences):
    result = []
    for sequence in sequences:
        events = [Evidence.model_validate(e) for e in sequence]
        if not events or not all(e.verified for e in events):
            raise ValueError('training requires nonempty verified episodes')
        task_ids = {events[0].before.task_id}
        for previous, current in zip(events, events[1:]):
            if previous.after.task_id == current.before.task_id:
                if previous.after != current.before:
                    raise ValueError('noncontiguous task history')
            elif not previous.after.done or current.before.spent:
                raise ValueError('new task before completion or without fresh state')
            else:
                if current.before.task_id in task_ids:
                    raise ValueError('duplicate task history')
                task_ids.add(current.before.task_id)
        result.append(events)
    if not result:
        raise ValueError('empty training data')
    return result


def batch_tensors(sequences):
    sequences = validated_sequences(sequences)
    shape = (len(sequences), max(map(len, sequences)))
    x = np.zeros((*shape, len(OBS_FEATURES)), np.float32)
    tokens = np.zeros((*shape, len(TOKEN_FEATURES)), np.float32)
    stats = np.zeros((*shape, STAT_DIM), np.float32)
    actions = np.zeros(shape, np.int64)
    returns = np.zeros(shape, np.float32)
    success = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.float32)
    for i, events in enumerate(sequences):
        history = HistoryStatistics()
        for j, event in enumerate(events):
            x[i, j] = observation_vector(event.before)
            tokens[i, j] = feedback_vector(event)
            stats[i, j] = history.vector()
            actions[i, j] = ACTIONS.index(event.action)
            success[i, j] = float(event.after.success)
            mask[i, j] = 1
            history.update(event)
        value = 0.
        for j in range(len(events)-1, -1, -1):
            if events[j].after.done:
                value = 0.
            value += events[j].reward
            returns[i, j] = value
    return {k: torch.from_numpy(v) for k, v in dict(x=x, tokens=tokens, stats=stats,
        actions=actions, returns=returns, success=success, mask=mask).items()}


class CognitiveNetwork(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        extra = STATE_DIM if kind == 'gru' else STAT_DIM if kind in {'context_mlp', 'linear'} else 0
        if kind == 'gru':
            self.recurrent = nn.GRU(len(TOKEN_FEATURES), STATE_DIM, batch_first=True)
        size = len(OBS_FEATURES) + extra
        # Match baseline parameter budgets before the study, not via test tuning:
        # GRU 4078, stateless MLP 4086, statistics MLP 4102 parameters.
        width = {'gru': 32, 'mlp': 204, 'context_mlp': 128, 'linear': 1}[kind]
        self.head = nn.Linear(size, 6) if kind == 'linear' else nn.Sequential(
            nn.Linear(size, width), nn.Tanh(), nn.Linear(width, 6))

    def heads(self, x, hidden, statistics):
        inputs = torch.cat((x, hidden), dim=-1) if self.kind == 'gru' else (
            torch.cat((x, statistics), dim=-1) if self.kind in {'context_mlp', 'linear'} else x)
        output = self.head(inputs)
        return output[..., :3], output[..., 3:]

    def forward(self, batch):
        x = batch['x']
        hidden = x.new_zeros((*x.shape[:2], STATE_DIM))
        if self.kind == 'gru':
            after, _ = self.recurrent(batch['tokens'])
            # Current choice can use only feedback preceding that choice.
            hidden = torch.cat((hidden[:, :1], after[:, :-1]), dim=1)
        return self.heads(x, hidden, batch['stats'])


class CognitiveSelector:
    def __init__(self, *, kind='gru', seed=0):
        if kind not in KINDS or type(seed) is not int:
            raise ValueError('unsupported model configuration')
        self.kind, self.seed = kind, seed
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = CognitiveNetwork(kind)
        self.history = []
        self.reset_context()

    def weights(self):
        return {k: v.detach().tolist() for k, v in self.model.state_dict().items()}

    def _restore_weights(self, weights):
        expected = self.model.state_dict()
        if not isinstance(weights, dict) or set(weights) != set(expected):
            raise ValueError('incompatible weights')
        restored = {}
        for key, value in weights.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            if tensor.shape != expected[key].shape or not torch.isfinite(tensor).all():
                raise ValueError('invalid weight shape or nonfinite weight')
            restored[key] = tensor
        self.model.load_state_dict(restored)

    @property
    def parameter_digest(self):
        return digest_state(dict(kind=self.kind, weights=self.weights()))

    @property
    def hidden_values(self):
        return self.hidden.flatten().tolist()

    def reset_context(self):
        self.hidden = torch.zeros(1, 1, STATE_DIM)
        self.statistics = HistoryStatistics()
        self.episodes = []
        self.observation = None

    def observe(self, observation):
        self.observation = PublicState.model_validate(observation)

    def all_predictions(self):
        if self.observation is None:
            raise RuntimeError('observe before suggest')
        with torch.no_grad():
            x = torch.tensor(observation_vector(self.observation), dtype=torch.float32)
            stats = torch.tensor(self.statistics.vector(), dtype=torch.float32)
            q, logits = self.model.heads(x, self.hidden.flatten(), stats)
        return q.tolist(), logits.sigmoid().tolist()

    def suggest(self, *, inquiry=True):
        q, probability = self.all_predictions()
        available = self.observation.available
        return sorted([dict(method=a, score=q[i], success_probability=probability[i])
            for i, a in enumerate(ACTIONS) if a in available and (inquiry or a != 'ask')],
            key=lambda c: (-c['score'], c['method']))

    def feedback(self, evidence):
        event = Evidence.model_validate(evidence)
        if not event.verified or event.before != self.observation:
            raise ValueError('feedback must be verified and match current observation')
        if self.episodes:
            validated_sequences([self.episodes+[event]])
        self._update_fast(event)
        self.episodes.append(event)
        self.observation = event.after

    def _update_fast(self, event):
        self.statistics.update(event)
        if self.kind == 'gru':
            with torch.no_grad():
                token = torch.tensor(feedback_vector(event), dtype=torch.float32).reshape(1, 1, -1)
                _, self.hidden = self.model.recurrent(token, self.hidden)

    def _reencode(self):
        self.hidden = torch.zeros(1, 1, STATE_DIM)
        self.statistics = HistoryStatistics()
        for event in self.episodes:
            self._update_fast(event)

    def network_predictions(self, batch):
        with torch.no_grad():
            q, logits = self.model(batch)
        return q.numpy(), logits.sigmoid().numpy()

    def learn(self, sequences, *, epochs=100):
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError('invalid training epoch count')
        batch = batch_tensors(sequences)  # validate before touching model state
        before, old = self.parameter_digest, self.weights()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.003)
        losses = []
        for epoch in range(epochs):
            q, logits = self.model(batch)
            action = batch['actions'][..., None]
            selected = q.gather(-1, action).squeeze(-1)
            selected_logits = logits.gather(-1, action).squeeze(-1)
            error = (selected-batch['returns']).square()
            classification = nn.functional.binary_cross_entropy_with_logits(
                selected_logits, batch['success'], reduction='none')
            loss = ((error+.25*classification)*batch['mask']).sum()/batch['mask'].sum()
            if not torch.isfinite(loss):
                self._restore_weights(old)
                raise FloatingPointError('nonfinite loss; previous parameters restored')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.history.append(old)
        new = self.weights()
        delta = sum(float(np.square(np.asarray(new[k])-np.asarray(old[k])).sum()) for k in old)**.5
        self._reencode()
        return dict(before=before, after=self.parameter_digest, parameter_delta=delta,
            verified_transitions=int(batch['mask'].sum()), epochs=epochs, losses=losses,
            parameters=sum(p.numel() for p in self.model.parameters()))

    def rollback(self):
        if not self.history:
            raise RuntimeError('no parameter update to roll back')
        self._restore_weights(self.history.pop())
        self._reencode()

    def save(self, path):
        data = dict(schema='cognitive-1', kind=self.kind, seed=self.seed,
            observation_features=list(OBS_FEATURES), token_features=list(TOKEN_FEATURES),
            weights=self.weights(), history=self.history,
            evidence=[e.model_dump(mode='json') for e in self.episodes],
            observation=self.observation.model_dump(mode='json') if self.observation else None)
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump(data, stream, allow_nan=False)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if (set(data) != {'schema', 'kind', 'seed', 'observation_features', 'token_features',
            'weights', 'history', 'evidence', 'observation'} or data['schema'] != 'cognitive-1'
            or data['observation_features'] != list(OBS_FEATURES)
            or data['token_features'] != list(TOKEN_FEATURES)):
            raise ValueError('incompatible cognitive checkpoint')
        result = cls(kind=data['kind'], seed=data['seed'])
        if not isinstance(data['history'], list):
            raise ValueError('invalid rollback history')
        for weights in data['history']:
            result._restore_weights(weights)
        result._restore_weights(data['weights'])
        result.history = data['history']
        events = [Evidence.model_validate(e) for e in data['evidence']]
        if events:
            validated_sequences([events])
        result.episodes = events
        result._reencode()
        if data['observation'] is not None:
            result.observe(data['observation'])
        return result
