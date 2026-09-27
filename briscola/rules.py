"""Game configuration (full or reduced deck) and the trick-taking rule."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .cards import RANKS, SUITS, Card


@dataclass(frozen=True)
class GameConfig:
    """Which cards are in the deck and how many cards each player holds.

    The default is standard two-player Briscola: 4 suits x 10 ranks, 3 cards
    per hand. Smaller decks (fewer suits and/or ranks) give reduced games that
    keep the same rules, which is what an equilibrium solver like CFR needs.
    Card points and strength always follow the standard tables, so a reduced
    deck with ranks (1, 3, 10) keeps Asso > Tre > Re worth 11/10/4.
    """

    suits: tuple[str, ...] = SUITS
    ranks: tuple[int, ...] = RANKS
    hand_size: int = 3

    def __post_init__(self) -> None:
        if not self.suits or len(set(self.suits)) != len(self.suits):
            raise ValueError("suits must be a non-empty tuple without duplicates")
        if not set(self.suits) <= set(SUITS):
            raise ValueError(f"unknown suit in {self.suits}; valid suits are {SUITS}")
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ranks must be a non-empty tuple without duplicates")
        if not set(self.ranks) <= set(RANKS):
            raise ValueError(f"unknown rank in {self.ranks}; valid ranks are {RANKS}")
        if self.hand_size < 1:
            raise ValueError("hand_size must be at least 1")
        n = len(self.suits) * len(self.ranks)
        # Two hands plus a face-up trump card must fit, and the stock (which
        # includes the trump card) is drawn two cards per trick.
        if n <= 2 * self.hand_size:
            raise ValueError(f"a deck of {n} cards is too small for hands of {self.hand_size}")
        if n % 2:
            raise ValueError(f"the deck must have an even number of cards, got {n}")

    @classmethod
    def standard(cls) -> GameConfig:
        return cls()

    @classmethod
    def reduced(
        cls,
        suits: tuple[str, ...] = ("Coppe", "Denari"),
        ranks: tuple[int, ...] = (1, 3, 10),
        hand_size: int = 2,
    ) -> GameConfig:
        """A small Briscola-like game; the default has 6 cards and 3 tricks."""
        return cls(suits=suits, ranks=ranks, hand_size=hand_size)

    @cached_property
    def deck(self) -> tuple[Card, ...]:
        """All cards in a fixed canonical order (suit-major)."""
        return tuple(Card(s, r) for s in self.suits for r in self.ranks)

    @cached_property
    def card_index(self) -> dict[Card, int]:
        """Card -> position in :attr:`deck`; used for one-hot encodings and action ids."""
        return {c: i for i, c in enumerate(self.deck)}

    @property
    def deck_size(self) -> int:
        return len(self.suits) * len(self.ranks)

    @property
    def num_tricks(self) -> int:
        return self.deck_size // 2

    @property
    def initial_stock_size(self) -> int:
        """Cards left to draw after the deal, including the face-up trump card."""
        return self.deck_size - 2 * self.hand_size

    @cached_property
    def total_points(self) -> int:
        return sum(c.points for c in self.deck)


STANDARD = GameConfig()


def trick_winner(lead: Card, follow: Card, trump_suit: str) -> int:
    """Return 0 if the lead card takes the trick, 1 if the follow card does.

    There is no obligation to follow suit. The follower wins only by playing a
    stronger card of the lead suit, or a trump on a non-trump lead.
    """
    if follow.suit == lead.suit:
        return 1 if follow.strength > lead.strength else 0
    return 1 if follow.suit == trump_suit else 0
