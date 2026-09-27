"""Play games between agents and evaluate them with duplicate deals.

For a command-line comparison see :mod:`briscola.compare`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .agents import Agent
from .cards import Card
from .game import GameState, IllegalMoveError
from .rules import STANDARD, GameConfig


@dataclass(frozen=True)
class GameResult:
    scores: tuple[int, int]
    winner: int | None
    returns: tuple[int, int]


def play_game(
    agents: Sequence[Agent],
    config: GameConfig = STANDARD,
    *,
    seed: int | str | None = None,
    deck: Sequence[Card] | None = None,
    first_player: int = 0,
) -> GameResult:
    """Play one game; ``agents[i]`` sits in seat ``i``. Reproducible given ``seed``."""
    if seed is None:
        seed = random.randrange(2**63)
    state = GameState.new_game(
        config, rng=random.Random(f"deck-{seed}"), deck=deck, first_player=first_player
    )
    rngs = [random.Random(f"agent{i}-{seed}") for i in (0, 1)]
    while not state.is_terminal:
        p = state.current_player
        card = agents[p].act(state.observation(p), rngs[p])
        if card not in state.hands[p]:
            raise IllegalMoveError(f"agent {agents[p].name} chose illegal card {card}")
        state.play(card)
    return GameResult(tuple(state.scores), state.winner(), state.returns())


@dataclass(frozen=True)
class MatchStats:
    """Results from the point of view of agent A."""

    agent_a: str
    agent_b: str
    games: int
    wins: int
    draws: int
    losses: int
    mean_reward: float
    reward_ci95: float
    mean_point_diff: float
    point_diff_ci95: float

    def __str__(self) -> str:
        return (
            f"{self.agent_a} vs {self.agent_b}: {self.games} partite\n"
            f"  vittorie {self.wins} ({self.wins / self.games:.1%}), "
            f"pareggi {self.draws} ({self.draws / self.games:.1%}), "
            f"sconfitte {self.losses} ({self.losses / self.games:.1%})\n"
            f"  reward medio {self.mean_reward:+.3f} ± {self.reward_ci95:.3f} (IC 95%)\n"
            f"  differenza punti media {self.mean_point_diff:+.2f} ± {self.point_diff_ci95:.2f}"
        )


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def evaluate(
    agent_a: Agent,
    agent_b: Agent,
    n_deals: int,
    config: GameConfig = STANDARD,
    *,
    seed: int = 0,
    duplicate: bool = True,
) -> MatchStats:
    """Compare two agents over ``n_deals`` shuffled decks.

    With ``duplicate=True`` every deck is played twice with the seats swapped,
    so both agents receive exactly the same cards and the same turn to lead.
    That removes most of the luck of the deal from the comparison. The
    confidence interval treats each pair of games as one sample, because the
    two games on the same deck are not independent.
    Without duplicate, the seats alternate from one deal to the next.
    """
    wins = draws = losses = 0
    rewards: list[float] = []
    diffs: list[float] = []
    for i in range(n_deals):
        deck = list(config.deck)
        random.Random(f"eval-{seed}-{i}").shuffle(deck)
        seatings = [0, 1] if duplicate else [i % 2]
        deal_rewards, deal_diffs = [], []
        for a_seat in seatings:
            agents = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
            result = play_game(agents, config, seed=f"{seed}-{i}-{a_seat}", deck=deck)
            r = result.returns[a_seat]
            wins += r == 1
            draws += r == 0
            losses += r == -1
            deal_rewards.append(r)
            deal_diffs.append(result.scores[a_seat] - result.scores[1 - a_seat])
        rewards.append(sum(deal_rewards) / len(deal_rewards))
        diffs.append(sum(deal_diffs) / len(deal_diffs))
    mean_r, ci_r = _mean_ci95(rewards)
    mean_d, ci_d = _mean_ci95(diffs)
    return MatchStats(
        agent_a.name, agent_b.name, wins + draws + losses,
        wins, draws, losses, mean_r, ci_r, mean_d, ci_d,
    )
