"""Play Briscola against a bot in the terminal: ``python -m briscola.play``."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence

from .agents import AGENTS, make_agent
from .game import GameState
from .rules import STANDARD, GameConfig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gioca a Briscola contro un bot.")
    parser.add_argument("--opponent", default="greedy", choices=sorted(AGENTS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--second", action="store_true", help="fai iniziare il bot")
    parser.add_argument("--reduced", action="store_true", help="usa il mazzo ridotto di default")
    args = parser.parse_args(argv)

    config = GameConfig.reduced() if args.reduced else STANDARD
    bot = make_agent(args.opponent)
    rng = random.Random(args.seed)
    human, cpu = (1, 0) if args.second else (0, 1)
    state = GameState.new_game(config, rng=rng, first_player=0)
    print(f"Briscola: {state.trump_card}\n")

    while not state.is_terminal:
        n_tricks = len(state.tricks)
        if state.current_player == cpu:
            card = bot.act(state.observation(cpu), rng)
            print(f"Il bot gioca: {card}")
        else:
            obs = state.observation(human)
            if obs.table:
                print(f"Sul tavolo: {obs.table[0]}")
            print(f"Punti: tu {obs.my_score} - bot {obs.opponent_score}   "
                  f"carte nel mazzo: {obs.stock_size}")
            for i, c in enumerate(obs.hand, 1):
                print(f"  {i}) {c}")
            card = obs.hand[_ask_index(len(obs.hand)) - 1]
            print(f"Giochi: {card}")
        state.play(card)
        if len(state.tricks) > n_tricks:
            trick = state.tricks[-1]
            who = "tu" if trick.winner == human else "il bot"
            print(f"-> Prende {who} ({trick.points} punti)\n")

    print(f"Fine: tu {state.scores[human]} - bot {state.scores[cpu]}")
    w = state.winner()
    print("Pareggio!" if w is None else ("Hai vinto!" if w == human else "Ha vinto il bot."))


def _ask_index(n: int) -> int:
    while True:
        raw = input(f"Quale carta giochi? [1-{n}] ").strip()
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw)
        print("Scelta non valida.")


if __name__ == "__main__":
    main()
