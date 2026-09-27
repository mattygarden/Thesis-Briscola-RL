"""Full game state and transition rules of two-player Briscola."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .cards import Card
from .observation import Observation, Trick
from .rules import STANDARD, GameConfig, trick_winner


class IllegalMoveError(ValueError):
    pass


class GameState:
    """The complete (omniscient) state of a game between players 0 and 1.

    Agents must not read it directly: they receive :meth:`observation`, which
    hides the opponent's hand and the stock order. Solvers and tests may use
    the full state, e.g. to enumerate deals or check invariants.

    Flow of a game:

    * The shuffled deck is dealt alternately, ``hand_size`` cards each,
      starting with ``first_player``, who also leads the first trick.
    * The next card is the face-up trump card (*briscola*); it goes under the
      stock and is the last card drawn.
    * The leader plays a card, then the other player responds. The trick
      winner captures both cards and leads the next trick.
    * While the stock lasts, the trick winner draws first, then the loser.
    * When all cards are played, the player with more points wins.
    """

    __slots__ = (
        "config", "hands", "stock", "trump_card", "first_player", "leader",
        "current_player", "table", "captured", "scores", "plays", "tricks",
        "draws", "initial_hands",
    )

    def __init__(self, config: GameConfig, deck: Sequence[Card], first_player: int = 0) -> None:
        if first_player not in (0, 1):
            raise ValueError("first_player must be 0 or 1")
        if len(deck) != config.deck_size or set(deck) != set(config.deck):
            raise ValueError("deck must be a permutation of config.deck")
        h = config.hand_size
        self.config = config
        self.hands: list[list[Card]] = [[], []]
        seats = (first_player, 1 - first_player)
        for i in range(2 * h):
            self.hands[seats[i % 2]].append(deck[i])
        self.trump_card = deck[2 * h]
        # Top of the stock is index 0; the face-up trump card is drawn last.
        self.stock: list[Card] = list(deck[2 * h + 1:]) + [self.trump_card]
        self.first_player = first_player
        self.leader = first_player
        self.current_player = first_player
        self.table: list[Card] = []
        self.captured: list[list[Card]] = [[], []]
        self.scores = [0, 0]
        self.plays: list[tuple[int, Card]] = []
        self.tricks: list[Trick] = []
        self.draws: list[tuple[int, Card]] = []
        self.initial_hands = (tuple(sorted(self.hands[0])), tuple(sorted(self.hands[1])))

    @classmethod
    def new_game(
        cls,
        config: GameConfig = STANDARD,
        *,
        seed: int | str | None = None,
        rng: random.Random | None = None,
        deck: Sequence[Card] | None = None,
        first_player: int = 0,
    ) -> GameState:
        """Start a game from an explicit ``deck`` order, or shuffle one with ``rng``/``seed``."""
        if deck is None:
            rng = rng if rng is not None else random.Random(seed)
            shuffled = list(config.deck)
            rng.shuffle(shuffled)
            deck = shuffled
        return cls(config, deck, first_player)

    @property
    def trump_suit(self) -> str:
        return self.trump_card.suit

    @property
    def is_terminal(self) -> bool:
        return len(self.tricks) == self.config.num_tricks

    def legal_actions(self) -> list[Card]:
        if self.is_terminal:
            return []
        return list(self.hands[self.current_player])

    def play(self, card: Card) -> None:
        """Play ``card`` for the player to move and advance the game."""
        if self.is_terminal:
            raise IllegalMoveError("the game is over")
        p = self.current_player
        hand = self.hands[p]
        if card not in hand:
            raise IllegalMoveError(f"player {p} does not hold {card}")
        hand.remove(card)
        self.plays.append((p, card))

        if not self.table:
            self.table.append(card)
            self.current_player = 1 - p
            return

        lead = self.table.pop()
        leader = self.leader
        winner = leader if trick_winner(lead, card, self.trump_suit) == 0 else p
        points = lead.points + card.points
        self.captured[winner] += (lead, card)
        self.scores[winner] += points
        self.tricks.append(Trick(leader, lead, card, winner, points))

        if self.stock:
            for q in (winner, 1 - winner):
                drawn = self.stock.pop(0)
                self.hands[q].append(drawn)
                self.draws.append((q, drawn))

        self.leader = winner
        self.current_player = winner

    def winner(self) -> int | None:
        """0 or 1 for the player with more points, None for a draw (e.g. 60-60)."""
        if not self.is_terminal:
            raise ValueError("the game is not over")
        if self.scores[0] == self.scores[1]:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1

    def returns(self) -> tuple[int, int]:
        """Zero-sum terminal payoffs: +1 win, 0 draw, -1 loss for (player 0, player 1)."""
        w = self.winner()
        if w is None:
            return (0, 0)
        return (1, -1) if w == 0 else (-1, 1)

    def observation(self, player: int) -> Observation:
        opponent = 1 - player
        return Observation(
            config=self.config,
            player=player,
            to_move=self.current_player,
            hand=tuple(self.hands[player]),
            trump_card=self.trump_card,
            table=tuple(self.table),
            scores=(self.scores[0], self.scores[1]),
            stock_size=len(self.stock),
            # The trump card is never dealt, so the opponent can only hold it
            # by having visibly drawn it as the last card of the stock.
            known_opponent_cards=(
                (self.trump_card,) if self.trump_card in self.hands[opponent] else ()
            ),
            plays=tuple(self.plays),
            tricks=tuple(self.tricks),
            initial_hand=self.initial_hands[player],
            my_draws=tuple(c for q, c in self.draws if q == player),
        )

    def clone(self) -> GameState:
        """An independent copy, for tree search (Card objects are immutable and shared)."""
        new = GameState.__new__(GameState)
        new.config = self.config
        new.hands = [list(self.hands[0]), list(self.hands[1])]
        new.stock = list(self.stock)
        new.trump_card = self.trump_card
        new.first_player = self.first_player
        new.leader = self.leader
        new.current_player = self.current_player
        new.table = list(self.table)
        new.captured = [list(self.captured[0]), list(self.captured[1])]
        new.scores = list(self.scores)
        new.plays = list(self.plays)
        new.tricks = list(self.tricks)
        new.draws = list(self.draws)
        new.initial_hands = self.initial_hands
        return new

    def render(self) -> str:
        """A human-readable, omniscient dump of the state (for debugging)."""
        def cards(cs):
            return " ".join(c.short for c in cs) or "-"
        return "\n".join([
            f"Briscola: {self.trump_card}   mazzo: {len(self.stock)} carte   "
            f"presa {len(self.tricks) + 1}/{self.config.num_tricks}",
            f"G0 mano: {cards(self.hands[0])}   punti: {self.scores[0]}",
            f"G1 mano: {cards(self.hands[1])}   punti: {self.scores[1]}",
            f"Tavolo: {cards(self.table)}   tocca a: G{self.current_player}",
        ])
