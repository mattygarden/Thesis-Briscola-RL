"""Compare two agents from the terminal.

    python -m briscola.compare --a greedy --b random --deals 5000
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from .agents import AGENTS, make_agent
from .arena import evaluate
from .rules import STANDARD, GameConfig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Confronta due agenti di Briscola.")
    parser.add_argument("--a", default="greedy", choices=sorted(AGENTS))
    parser.add_argument("--b", default="random", choices=sorted(AGENTS))
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-duplicate", action="store_true")
    parser.add_argument("--reduced", action="store_true", help="usa il mazzo ridotto di default")
    args = parser.parse_args(argv)

    config = GameConfig.reduced() if args.reduced else STANDARD
    start = time.perf_counter()
    stats = evaluate(
        make_agent(args.a), make_agent(args.b), args.deals, config,
        seed=args.seed, duplicate=not args.no_duplicate,
    )
    elapsed = time.perf_counter() - start
    print(stats)
    print(f"  {stats.games / elapsed:,.0f} partite/s ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
