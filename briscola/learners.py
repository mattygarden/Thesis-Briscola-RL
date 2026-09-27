"""Learning agents. The reference learner is linear Q-learning in pure Python.

It is deliberately simple: it validates the training pipeline end to end and
gives a first, interpretable baseline for the memory experiment. Neural
learners (DQN, PPO) can replace it without changing the environment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .cards import Card
from .encoding import action_mask, encode, feature_size
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
