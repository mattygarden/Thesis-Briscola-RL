"""Learning agents: linear Q-learning in pure Python, in two flavours.

* :class:`LinearQLearner`: one weight vector per card over the state vector
  of :mod:`briscola.encoding` (conditions "basic" and "memory").
* :class:`FeatureQLearner`: one shared weight vector over the per-card
  features of :mod:`briscola.features` (condition "features").

Both expose the same interface, used by :func:`briscola.train.train`:
``featurize(obs)`` turns an observation into the learner's input ``x``, then
``select_action(x, mask, rng, epsilon)`` and ``update(x, a, r, x_next,
mask_next, done)``. Neural learners (DQN, PPO) can implement the same methods.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .cards import Card
from .encoding import action_mask, encode, feature_size
from .features import FEATURE_NAMES, NUM_FEATURES, action_features
from .observation import Observation
from .rules import STANDARD, GameConfig


def _sparse(x: list[float]) -> list[tuple[int, float]]:
    return [(i, v) for i, v in enumerate(x) if v]


class LinearQLearner:
    """Linear action values ``Q(s, a) = w_a · x(s) + b_a``, one weight vector per card.

    Trained with semi-gradient Q-learning: after each transition the weights
    of the chosen card move towards the target ``r + gamma * max_a' Q(s', a')``
    (only ``r`` at the end of the game). The step is normalised by the squared
    norm of the features (NLMS), so an encoding with more active features
    (the card-counting one) does not get a larger effective learning rate.

    Because each card has its own weights, the model can learn per-card
    rules such as "play the Tre di Coppe once the Asso di Coppe is gone",
    but not interactions between two features beyond that.

    It also implements the :class:`~briscola.agents.Agent` protocol (greedy
    play), so it can be evaluated with :func:`~briscola.arena.evaluate` or
    used as an opponent (self-play).
    """

    def __init__(
        self,
        config: GameConfig = STANDARD,
        memory: bool = True,
        *,
        lr: float = 0.2,
        gamma: float = 1.0,
        name: str | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.lr = lr
        self.gamma = gamma
        self.name = name or ("linear-q-memory" if memory else "linear-q-basic")
        self.n_features = feature_size(config, memory)
        self.w = [[0.0] * self.n_features for _ in range(config.deck_size)]
        self.b = [0.0] * config.deck_size

    def featurize(self, obs: Observation) -> list[float]:
        return encode(obs, self.memory)

    def q_value(self, xs: list[tuple[int, float]], action: int) -> float:
        w = self.w[action]
        return self.b[action] + sum(w[i] * v for i, v in xs)

    def greedy_action(self, x: list[float], mask: list[bool], rng: random.Random) -> int:
        xs = _sparse(x)
        best, best_q = [], float("-inf")
        for a, legal in enumerate(mask):
            if not legal:
                continue
            q = self.q_value(xs, a)
            if q > best_q + 1e-12:
                best, best_q = [a], q
            elif abs(q - best_q) <= 1e-12:
                best.append(a)
        return best[0] if len(best) == 1 else rng.choice(best)

    def select_action(
        self, x: list[float], mask: list[bool], rng: random.Random, epsilon: float = 0.0
    ) -> int:
        """Epsilon-greedy choice among the legal actions."""
        if epsilon > 0 and rng.random() < epsilon:
            return rng.choice([a for a, legal in enumerate(mask) if legal])
        return self.greedy_action(x, mask, rng)

    def update(
        self,
        x: list[float],
        action: int,
        reward: float,
        x_next: list[float],
        mask_next: list[bool],
        done: bool,
    ) -> float:
        """One Q-learning step; returns the TD error."""
        target = reward
        if not done:
            xs_next = _sparse(x_next)
            target += self.gamma * max(
                self.q_value(xs_next, a) for a, legal in enumerate(mask_next) if legal
            )
        xs = _sparse(x)
        td_error = target - self.q_value(xs, action)
        step = self.lr * td_error / (1.0 + sum(v * v for _, v in xs))
        w = self.w[action]
        for i, v in xs:
            w[i] += step * v
        self.b[action] += step
        return td_error

    def act(self, obs: Observation, rng: random.Random) -> Card:
        """Greedy play from a raw observation (Agent protocol)."""
        a = self.greedy_action(encode(obs, self.memory), action_mask(obs), rng)
        return self.config.deck[a]

    def save(self, path: str | Path) -> None:
        data = {
            "kind": "LinearQLearner",
            "name": self.name,
            "memory": self.memory,
            "lr": self.lr,
            "gamma": self.gamma,
            "config": {
                "suits": list(self.config.suits),
                "ranks": list(self.config.ranks),
                "hand_size": self.config.hand_size,
            },
            "w": self.w,
            "b": self.b,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> LinearQLearner:
        data = json.loads(Path(path).read_text())
        cfg = data["config"]
        config = GameConfig(tuple(cfg["suits"]), tuple(cfg["ranks"]), cfg["hand_size"])
        learner = cls(config, data["memory"], lr=data["lr"], gamma=data["gamma"], name=data["name"])
        learner.w, learner.b = data["w"], data["b"]
        return learner


class FeatureQLearner:
    """Linear Q-learning over per-card features: ``Q(s, a) = w · phi(s, a)``.

    The weights are shared by all cards, so the model has
    :data:`~briscola.features.NUM_FEATURES` parameters and what it learns
    about one card transfers to every other. The input ``x`` produced by
    :meth:`featurize` is a dict {deck index: phi(s, a)} over the legal cards.
    The update is the same NLMS-normalised semi-gradient Q-learning step as
    :class:`LinearQLearner`.
    """

    def __init__(
        self,
        config: GameConfig = STANDARD,
        *,
        lr: float = 0.2,
        gamma: float = 1.0,
        name: str = "linear-q-features",
    ) -> None:
        self.config = config
        self.lr = lr
        self.gamma = gamma
        self.name = name
        self.w = [0.0] * NUM_FEATURES

    def featurize(self, obs: Observation) -> dict[int, list[float]]:
        return action_features(obs)

    def q_value(self, phi: list[float]) -> float:
        return sum(wi * fi for wi, fi in zip(self.w, phi))

    def greedy_action(self, x: dict[int, list[float]], mask: list[bool], rng: random.Random) -> int:
        best, best_q = [], float("-inf")
        for a, phi in x.items():
            q = self.q_value(phi)
            if q > best_q + 1e-12:
                best, best_q = [a], q
            elif abs(q - best_q) <= 1e-12:
                best.append(a)
        return best[0] if len(best) == 1 else rng.choice(best)

    def select_action(
        self, x: dict[int, list[float]], mask: list[bool], rng: random.Random, epsilon: float = 0.0
    ) -> int:
        if epsilon > 0 and rng.random() < epsilon:
            return rng.choice(list(x))
        return self.greedy_action(x, mask, rng)

    def update(
        self,
        x: dict[int, list[float]],
        action: int,
        reward: float,
        x_next: dict[int, list[float]],
        mask_next: list[bool],
        done: bool,
    ) -> float:
        target = reward
        if not done and x_next:
            target += self.gamma * max(self.q_value(phi) for phi in x_next.values())
        phi = x[action]
        td_error = target - self.q_value(phi)
        step = self.lr * td_error / (1.0 + sum(f * f for f in phi))
        for i, f in enumerate(phi):
            self.w[i] += step * f
        return td_error

    def act(self, obs: Observation, rng: random.Random) -> Card:
        return self.config.deck[self.greedy_action(action_features(obs), [], rng)]

    def weights(self) -> dict[str, float]:
        """The learned weight of each named feature."""
        return dict(zip(FEATURE_NAMES, self.w))

    def save(self, path: str | Path) -> None:
        data = {
            "kind": "FeatureQLearner",
            "name": self.name,
            "lr": self.lr,
            "gamma": self.gamma,
            "config": {
                "suits": list(self.config.suits),
                "ranks": list(self.config.ranks),
                "hand_size": self.config.hand_size,
            },
            "features": list(FEATURE_NAMES),
            "w": self.w,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> FeatureQLearner:
        data = json.loads(Path(path).read_text())
        if data.get("features") != list(FEATURE_NAMES):
            raise ValueError("saved weights use a different feature set")
        cfg = data["config"]
        config = GameConfig(tuple(cfg["suits"]), tuple(cfg["ranks"]), cfg["hand_size"])
        learner = cls(config, lr=data["lr"], gamma=data["gamma"], name=data["name"])
        learner.w = data["w"]
        return learner
