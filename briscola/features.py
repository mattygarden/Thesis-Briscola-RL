"""Hand-crafted state-action features: a small vector per (situation, card) pair.

Instead of one long state vector, each legal card gets its own vector
phi(s, a) describing that card in that situation: how likely it is to be
beaten, how many points it risks, whether it takes the trick on the table,
and a few interactions with the context. A linear model with shared weights
over phi learns rules that apply to every card ("a safe card is good when..."),
with 16 parameters instead of thousands.

These features use the cards already played (through ``unseen_cards``), so
they are an *elaborated* form of card memory: we compute the inferences, the
model only learns how much to weigh them.
"""

from __future__ import annotations

from math import comb

from .cards import Card
from .observation import Observation
from .rules import trick_winner

FEATURE_NAMES: tuple[str, ...] = (
    "is_trump",                # 1  the card is a trump
    "points",                  # 2  card points / 11
    "strength",                # 3  strength within its suit / 9
    "p_not_beaten",            # 4  leading: P(opponent cannot beat it); following: wins_trick
    "top_of_suit",             # 5  no stronger card of its suit is still unseen
    "wins_trick",              # 6  following: it takes the card on the table
    "trick_points_if_win",     # 7  following and winning: points captured / 22
    "trump_x_stock",           # 8  is_trump * stock left / initial stock
    "points_x_leading",        # 9  points * leading
    "points_at_risk",          # 10 points * (1 - p_not_beaten)
    "trump_x_table_points",    # 11 is_trump * points on the table / 11
    "trump_x_trumps_unseen",   # 12 is_trump * trumps not yet seen / suit size
    "wins_x_points_needed",    # 13 wins_trick * points still needed to win / threshold
    "points_x_points_left",    # 14 points * points still out of sight / total
    "leading_x_p_not_beaten",  # 15 leading * p_not_beaten
    "bias",                    # 16 constant
)
NUM_FEATURES = len(FEATURE_NAMES)


def _beats(challenger: Card, card: Card, trump_suit: str) -> bool:
    """True if ``challenger``, played after ``card``, takes the trick."""
    return trick_winner(card, challenger, trump_suit) == 1


def p_not_beaten(card: Card, unseen: frozenset[Card], known_opponent: tuple[Card, ...],
                 opponent_unknown: int, trump_suit: str) -> float:
    """Probability that the opponent holds no card able to beat ``card``.

    Assumes the opponent's unknown cards are a uniformly random subset of the
    unseen cards (the rest are in the stock). A known opponent card that beats
    ``card`` makes the probability 0.
    """
    if any(_beats(c, card, trump_suit) for c in known_opponent):
        return 0.0
    u = len(unseen)
    if opponent_unknown <= 0 or u == 0:
        return 1.0
    k = sum(1 for c in unseen if _beats(c, card, trump_suit))
    return comb(u - k, opponent_unknown) / comb(u, opponent_unknown)


def action_features(obs: Observation) -> dict[int, list[float]]:
    """phi(s, a) for every legal card, keyed by the card's deck index."""
    legal = obs.legal_actions
    if not legal:
        return {}
    cfg = obs.config
    ts = obs.trump_suit
    unseen = obs.unseen_cards()
    known = obs.known_opponent_cards
    leading = not obs.table
    # The opponent holds as many cards as we do, one fewer if it already led.
    opp_hand = len(obs.hand) - (0 if leading else 1)
    opp_unknown = opp_hand - len(known)
    out_of_sight = list(unseen) + list(known)
    points_left = sum(c.points for c in out_of_sight) / cfg.total_points
    trumps_unseen = sum(1 for c in out_of_sight if c.suit == ts) / len(cfg.ranks)
    stock_frac = obs.stock_size / cfg.initial_stock_size
    threshold = cfg.total_points / 2
    needed = max(0.0, threshold + 0.5 - obs.my_score) / threshold
    table_points = obs.table[0].points if obs.table else 0

    features: dict[int, list[float]] = {}
    for card in legal:
        is_trump = 1.0 if card.suit == ts else 0.0
        pts = card.points / 11
        top = 0.0 if any(c.suit == card.suit and c.strength > card.strength
                         for c in out_of_sight) else 1.0
        if leading:
            wins, trick_pts = 0.0, 0.0
            p_safe = p_not_beaten(card, unseen, known, opp_unknown, ts)
        else:
            # Following closes the trick: the outcome is certain.
            wins = 1.0 if trick_winner(obs.table[0], card, ts) == 1 else 0.0
            trick_pts = wins * (table_points + card.points) / 22
            p_safe = wins
        lead = 1.0 if leading else 0.0
        features[cfg.card_index[card]] = [
            is_trump,
            pts,
            card.strength / 9,
            p_safe,
            top,
            wins,
            trick_pts,
            is_trump * stock_frac,
            pts * lead,
            pts * (1.0 - p_safe),
            is_trump * table_points / 11,
            is_trump * trumps_unseen,
            wins * needed,
            pts * points_left,
            lead * p_safe,
            1.0,
        ]
    return features
