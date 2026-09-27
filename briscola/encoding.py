"""Fixed-size feature vectors and action ids for reinforcement-learning agents.

Two observation encodings match the two conditions of the memory experiment:

* ``memory=False`` (basic): hand, table, trump, scores, stock size. Scores and
  stock size still summarise part of the history, so this agent is "without
  explicit card memory", not memoryless.
* ``memory=True`` (card-counting): the basic features plus the set of cards
  already played.

Vectors are plain lists of floats (``numpy.asarray`` / ``torch.tensor`` turn
them into arrays). An action is the index of a card in ``config.deck``.
"""

from __future__ import annotations

from collections.abc import Iterable

from .cards import Card
from .observation import Observation
from .rules import GameConfig


def one_hot_cards(cards: Iterable[Card], config: GameConfig) -> list[float]:
    vec = [0.0] * config.deck_size
    for c in cards:
        vec[config.card_index[c]] = 1.0
    return vec


def feature_size(config: GameConfig, memory: bool) -> int:
    n, s = config.deck_size, len(config.suits)
    size = 4 * n + s + 4  # hand, table, trump card, known opponent cards; trump suit; scalars
    return size + n if memory else size


def encode(obs: Observation, memory: bool) -> list[float]:
    cfg = obs.config
    completed = [c for t in obs.tricks for c in (t.lead, t.follow)]
    features = (
        one_hot_cards(obs.hand, cfg)
        + one_hot_cards(obs.table, cfg)
        + one_hot_cards([obs.trump_card], cfg)
        + one_hot_cards(obs.known_opponent_cards, cfg)
        + [1.0 if s == obs.trump_suit else 0.0 for s in cfg.suits]
        + [
            obs.my_score / cfg.total_points,
            obs.opponent_score / cfg.total_points,
            obs.stock_size / cfg.initial_stock_size,
            1.0 if obs.is_leading else 0.0,
        ]
    )
    if memory:
        features += one_hot_cards(completed, cfg)
    return features


def encode_basic(obs: Observation) -> list[float]:
    return encode(obs, memory=False)


def encode_card_counting(obs: Observation) -> list[float]:
    return encode(obs, memory=True)


def action_mask(obs: Observation) -> list[bool]:
    """True at the deck index of every legal card."""
    mask = [False] * obs.config.deck_size
    for c in obs.legal_actions:
        mask[obs.config.card_index[c]] = True
    return mask


def card_to_action(card: Card, config: GameConfig) -> int:
    return config.card_index[card]


def action_to_card(action: int, config: GameConfig) -> Card:
    return config.deck[action]
