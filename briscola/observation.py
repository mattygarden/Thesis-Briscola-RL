"""What one player can legitimately know at a given point of the game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .cards import Card
from .rules import GameConfig


class Trick(NamedTuple):
    """A completed trick. Both cards are face-up, so tricks are public information."""

    leader: int
    lead: Card
    follow: Card
    winner: int
    points: int


@dataclass(frozen=True)
class Observation:
    """A player's view of the game: everything visible to them, nothing else.

    It never contains the opponent's hand or the stock order. The only
    opponent card it can reveal is the face-up trump card, once the opponent
    has drawn it (everyone saw who took it) and until they play it.
    """

    config: GameConfig
    player: int
    to_move: int
    hand: tuple[Card, ...]
    trump_card: Card
    table: tuple[Card, ...]
    """The lead card of the current trick, if the opponent has already led."""
    scores: tuple[int, int]
    """Points captured by player 0 and player 1 (derivable from the public tricks)."""
    stock_size: int
    """Cards still to be drawn, including the face-up trump card while it is there."""
    known_opponent_cards: tuple[Card, ...]
    plays: tuple[tuple[int, Card], ...]
    """Every card played so far, in order, as (player, card); includes the table card."""
    tricks: tuple[Trick, ...]
    initial_hand: tuple[Card, ...]
    my_draws: tuple[Card, ...]
    """Cards this player drew from the stock, in order (private information)."""

    @property
    def trump_suit(self) -> str:
        return self.trump_card.suit

    @property
    def opponent(self) -> int:
        return 1 - self.player

    @property
    def is_my_turn(self) -> bool:
        return self.to_move == self.player

    @property
    def is_leading(self) -> bool:
        """True if it is this player's turn and the table is empty."""
        return self.is_my_turn and not self.table

    @property
    def my_score(self) -> int:
        return self.scores[self.player]

    @property
    def opponent_score(self) -> int:
        return self.scores[self.opponent]

    @property
    def legal_actions(self) -> tuple[Card, ...]:
        """Any card in hand may be played (no obligation to follow suit)."""
        return self.hand if self.is_my_turn else ()

    @property
    def played_cards(self) -> frozenset[Card]:
        """All cards seen on the table so far, including the current lead card."""
        return frozenset(card for _, card in self.plays)

    def unseen_cards(self) -> frozenset[Card]:
        """Cards whose location this player does not know.

        They are split between the opponent's hand and the stock. Once the
        stock is empty they are exactly the opponent's hand (minus
        ``known_opponent_cards``), which is how a card-counting player can
        infer everything in the last tricks.
        """
        seen = set(self.hand) | self.played_cards | {self.trump_card}
        return frozenset(c for c in self.config.deck if c not in seen)

    def info_state_key(self) -> tuple:
        """A hashable key identifying this player's information set.

        Two game states share a key exactly when this player cannot tell them
        apart. The key holds everything the player has observed: the initial
        hand, the trump card, the public sequence of plays and their own draws.
        Draws always happen right after each trick, so the two sequences
        together fix the full order of events: the key has perfect recall,
        as CFR's convergence guarantee requires.
        """
        return (self.player, self.initial_hand, self.trump_card, self.plays, self.my_draws)
