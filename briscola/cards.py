"""Cards of the Italian 40-card deck (semi italiani) and their Briscola values."""

from __future__ import annotations

from typing import NamedTuple

SUITS: tuple[str, ...] = ("Coppe", "Denari", "Bastoni", "Spade")
RANKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

RANK_NAMES: dict[int, str] = {
    1: "Asso", 2: "Due", 3: "Tre", 4: "Quattro", 5: "Cinque",
    6: "Sei", 7: "Sette", 8: "Fante", 9: "Cavallo", 10: "Re",
}
RANK_SYMBOLS: dict[int, str] = {
    1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "F", 9: "C", 10: "R",
}

# Card points: Asso 11, Tre 10, Re 4, Cavallo 3, Fante 2, all others 0 (total 120).
POINTS: dict[int, int] = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}

# Taking power within a suit, from weakest to strongest:
# 2 < 4 < 5 < 6 < 7 < Fante < Cavallo < Re < Tre < Asso.
STRENGTH: dict[int, int] = {2: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 3: 8, 1: 9}


class Card(NamedTuple):
    """An immutable, hashable card, e.g. ``Card("Denari", 1)`` is the Asso di Denari."""

    suit: str
    rank: int

    @property
    def points(self) -> int:
        return POINTS[self.rank]

    @property
    def strength(self) -> int:
        return STRENGTH[self.rank]

    @property
    def short(self) -> str:
        """Compact label: rank symbol + suit initial, e.g. ``"Ad"``, ``"Rc"``, ``"3s"``."""
        return RANK_SYMBOLS[self.rank] + self.suit[0].lower()

    def __str__(self) -> str:
        return f"{RANK_NAMES[self.rank]} di {self.suit}"
