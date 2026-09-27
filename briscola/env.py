"""Single-agent training environment: one learning agent against fixed opponents.

The API mirrors Gymnasium (``reset`` returns ``(obs, info)``, ``step`` returns
``(obs, reward, terminated, truncated, info)``) without depending on it, so a
Gymnasium wrapper is a few lines. Opponents are part of the environment: after
each agent action, ``step`` plays the opponent's cards (and the draws) until it
is the agent's turn again or the game ends.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .agents import Agent
from .encoding import action_mask, encode, feature_size
from .game import GameState, IllegalMoveError
from .observation import Observation
from .rules import STANDARD, GameConfig

REWARD_MODES = ("win", "points", "trick_points")


class BriscolaEnv:
    """Two-player Briscola seen from one learning agent.

    Parameters
    ----------
    opponents:
        One agent, or several: each episode draws one uniformly at random
        (a fixed mixture of opponents).
    config:
        Full game (default) or a reduced deck.
    memory:
        Observation encoding: ``False`` = basic, ``True`` = with the set of
        cards already played (see :mod:`briscola.encoding`).
    agent_seat:
        0 or 1 to fix the agent's seat (seat 0 leads the first trick), or
        ``None`` to draw it at random each episode.
    reward:
        ``"win"``: +1 / 0 / -1 at the end of the game (the thesis objective).
        ``"points"``: final point difference / total points, at the end.
        ``"trick_points"``: the same difference paid out trick by trick, so
        the rewards of an episode sum to the ``"points"`` reward.
    seed:
        Seeds the environment's random generator (deals, seats, opponents).
    """

    def __init__(
        self,
        opponents: Agent | Sequence[Agent],
        config: GameConfig = STANDARD,
        *,
        memory: bool = True,
        agent_seat: int | None = None,
        reward: str = "win",
        seed: int | str | None = None,
    ) -> None:
        self.opponents = [opponents] if hasattr(opponents, "act") else list(opponents)
        if not self.opponents:
            raise ValueError("at least one opponent is required")
        if agent_seat not in (None, 0, 1):
            raise ValueError("agent_seat must be 0, 1 or None")
        if reward not in REWARD_MODES:
            raise ValueError(f"reward must be one of {REWARD_MODES}")
        self.config = config
        self.memory = memory
        self.agent_seat = agent_seat
        self.reward_mode = reward
        self.rng = random.Random(seed)
        self.state: GameState | None = None
        self.seat = 0
        self.opponent: Agent = self.opponents[0]
        self._last_diff = 0

    @property
    def observation_size(self) -> int:
        return feature_size(self.config, self.memory)

    @property
    def action_size(self) -> int:
        """One action per card of the deck; ``action_mask`` marks the legal ones."""
        return self.config.deck_size

    def reset(self, seed: int | str | None = None) -> tuple[list[float], dict]:
        """Start a new game and return the agent's first observation."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.seat = self.agent_seat if self.agent_seat is not None else self.rng.randrange(2)
        self.opponent = self.rng.choice(self.opponents)
        self.state = GameState.new_game(self.config, rng=self.rng, first_player=0)
        self._last_diff = 0
        self._play_opponent()
        return self._encode(), self._info()

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict]:
        """Play the card with deck index ``action``, then the opponent's replies."""
        state = self.state
        if state is None or state.is_terminal:
            raise RuntimeError("call reset() before step()")
        card = self.config.deck[action]
        if card not in state.hands[self.seat]:
            raise IllegalMoveError(f"action {action} ({card}) is not in the agent's hand")
        state.play(card)
        self._play_opponent()
        return self._encode(), self._reward(), state.is_terminal, False, self._info()

    def observation(self) -> Observation:
        """The raw :class:`Observation` behind the current feature vector."""
        return self.state.observation(self.seat)

    def action_mask(self) -> list[bool]:
        return action_mask(self.observation())

    def _play_opponent(self) -> None:
        state, opp = self.state, 1 - self.seat
        while not state.is_terminal and state.current_player == opp:
            card = self.opponent.act(state.observation(opp), self.rng)
            if card not in state.hands[opp]:
                raise IllegalMoveError(f"opponent {self.opponent.name} chose illegal card {card}")
            state.play(card)

    def _reward(self) -> float:
        state = self.state
        if self.reward_mode == "trick_points":
            diff = state.scores[self.seat] - state.scores[1 - self.seat]
            r, self._last_diff = diff - self._last_diff, diff
            return r / self.config.total_points
        if not state.is_terminal:
            return 0.0
        if self.reward_mode == "win":
            return float(state.returns()[self.seat])
        return (state.scores[self.seat] - state.scores[1 - self.seat]) / self.config.total_points

    def _encode(self) -> list[float]:
        return encode(self.observation(), self.memory)

    def _info(self) -> dict:
        state = self.state
        info = {"action_mask": self.action_mask(), "seat": self.seat, "opponent": self.opponent.name}
        if state.is_terminal:
            info["scores"] = (state.scores[self.seat], state.scores[1 - self.seat])
            info["outcome"] = state.returns()[self.seat]
        return info
