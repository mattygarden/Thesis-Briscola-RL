"""Baseline agents: random play and two transparent heuristics."""

from __future__ import annotations

import random
from typing import Protocol

from .cards import Card
from .observation import Observation
from .rules import trick_winner


class Agent(Protocol):
    name: str

    def act(self, obs: Observation, rng: random.Random) -> Card:
        """Choose one of ``obs.legal_actions``."""
        ...


def _discard_key(card: Card, trump_suit: str) -> tuple[bool, int, int]:
    """Sort key for the cheapest card to give away: non-trump, few points, weak."""
    return (card.suit == trump_suit, card.points, card.strength)


class RandomAgent:
    """Plays a uniformly random legal card. A sanity-check baseline."""

    name = "random"

    def act(self, obs: Observation, rng: random.Random) -> Card:
        return rng.choice(obs.legal_actions)


class LowestCardAgent:
    """Always throws its cheapest card and never tries to win a trick on purpose."""

    name = "lowest"

    def act(self, obs: Observation, rng: random.Random) -> Card:
        return min(obs.legal_actions, key=lambda c: _discard_key(c, obs.trump_suit))


class GreedyAgent:
    """A simple one-trick-lookahead heuristic.

    Leading: play the cheapest card (keep trumps and points).
    Following:
      1. if a card of the lead suit wins (non-trump lead), take the trick with
         the one worth the most points: the trick is closed, so this is safe;
      2. else if the lead card is an Asso or a Tre, take it with the weakest
         winning trump;
      3. else throw the cheapest card.
    """

    name = "greedy"

    def act(self, obs: Observation, rng: random.Random) -> Card:
        hand, ts = obs.legal_actions, obs.trump_suit
        if not obs.table:
            return min(hand, key=lambda c: _discard_key(c, ts))
        lead = obs.table[0]
        winners = [c for c in hand if trick_winner(lead, c, ts) == 1]
        if lead.suit != ts:
            same_suit = [c for c in winners if c.suit == lead.suit]
            if same_suit:
                return max(same_suit, key=lambda c: (c.points, -c.strength))
        if winners and lead.points >= 10:
            return min(winners, key=lambda c: (c.strength, c.points))
        return min(hand, key=lambda c: _discard_key(c, ts))


AGENTS: dict[str, type] = {
    RandomAgent.name: RandomAgent,
    LowestCardAgent.name: LowestCardAgent,
    GreedyAgent.name: GreedyAgent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENTS[name]()
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}") from None
