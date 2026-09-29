"""Training loop with periodic evaluation, and its command-line interface.

    python -m briscola.train --memory --episodes 50000 --out runs/memory-s0
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .agents import AGENTS, Agent, make_agent
from .arena import evaluate
from .env import REWARD_MODES, BriscolaEnv
from .learners import FeatureQLearner, LinearQLearner
from .rules import STANDARD, GameConfig

EVAL_SEED = 10_000
"""Evaluation deals use their own seed, so they never coincide with training deals by design."""


def epsilon_at(episode: int, total: int, start: float, end: float, decay_fraction: float) -> float:
    """Linear decay from ``start`` to ``end`` over the first ``decay_fraction`` of training."""
    decay_episodes = max(1, int(total * decay_fraction))
    frac = min(1.0, episode / decay_episodes)
    return start + frac * (end - start)


def train(
    learner: LinearQLearner | FeatureQLearner,
    env: BriscolaEnv,
    episodes: int,
    *,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.02,
    decay_fraction: float = 0.8,
    eval_every: int = 0,
    eval_opponents: Sequence[Agent] = (),
    eval_deals: int = 500,
    seed: int = 0,
    on_eval: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Train ``learner`` in ``env`` for ``episodes`` games.

    Every ``eval_every`` episodes (and at the end) the greedy policy is
    evaluated against each of ``eval_opponents`` on held-out duplicate deals.
    Returns one row per evaluation point and opponent.
    """
    rng = random.Random(f"learner-{seed}")
    env.rng = random.Random(f"env-{seed}")
    history: list[dict] = []
    start = time.perf_counter()
    outcomes: list[float] = []

    def run_eval(episode: int, epsilon: float) -> None:
        train_avg = sum(outcomes) / len(outcomes) if outcomes else float("nan")
        outcomes.clear()
        for opp in eval_opponents:
            stats = evaluate(learner, opp, eval_deals, env.config, seed=EVAL_SEED)
            row = {
                "episode": episode,
                "epsilon": round(epsilon, 4),
                "opponent": opp.name,
                "mean_reward": stats.mean_reward,
                "reward_ci95": stats.reward_ci95,
                "win_rate": stats.wins / stats.games,
                "mean_point_diff": stats.mean_point_diff,
                "train_outcome_avg": train_avg,
                "elapsed_s": round(time.perf_counter() - start, 1),
            }
            history.append(row)
            if on_eval:
                on_eval(row)

    if eval_every and eval_opponents:
        run_eval(0, epsilon_start)
    for ep in range(1, episodes + 1):
        epsilon = epsilon_at(ep - 1, episodes, epsilon_start, epsilon_end, decay_fraction)
        _, info = env.reset()
        x = learner.featurize(env.observation())
        done = False
        while not done:
            a = learner.select_action(x, info["action_mask"], rng, epsilon)
            _, reward, done, _, info = env.step(a)
            x_next = learner.featurize(env.observation())
            learner.update(x, a, reward, x_next, info["action_mask"], done)
            x = x_next
        outcomes.append(info["outcome"])
        if eval_opponents and ((eval_every and ep % eval_every == 0) or ep == episodes):
            if not history or history[-1]["episode"] != ep:
                run_eval(ep, epsilon)
    return history


def write_history(history: list[dict], path: str | Path) -> None:
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _agents(names: str) -> list[Agent]:
    return [make_agent(n.strip()) for n in names.split(",") if n.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Addestra un agente Q-learning lineare (stato o feature per carta).")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True,
                        help="encoding con le carte già giocate (default) o --no-memory")
    parser.add_argument("--features", action="store_true",
                        help="usa FeatureQLearner (feature per carta) invece del learner lineare")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--opponents", default="random,lowest,greedy",
                        help=f"avversari di addestramento, separati da virgole: {sorted(AGENTS)}")
    parser.add_argument("--eval-opponents", default="random,lowest,greedy")
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-deals", type=int, default=500)
    parser.add_argument("--reward", default="trick_points", choices=REWARD_MODES)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reduced", action="store_true", help="usa il mazzo ridotto di default")
    parser.add_argument("--out", default=None, help="cartella dove salvare pesi e storico")
    args = parser.parse_args(argv)

    config = GameConfig.reduced() if args.reduced else STANDARD
    env = BriscolaEnv(_agents(args.opponents), config, memory=args.memory, reward=args.reward)
    if args.features:
        learner = FeatureQLearner(config, lr=args.lr)
    else:
        learner = LinearQLearner(config, args.memory, lr=args.lr)

    def show(row: dict) -> None:
        print(f"ep {row['episode']:>7}  eps {row['epsilon']:.3f}  vs {row['opponent']:<7} "
              f"reward {row['mean_reward']:+.3f} ± {row['reward_ci95']:.3f}  "
              f"vittorie {row['win_rate']:.1%}  [{row['elapsed_s']} s]", flush=True)

    history = train(
        learner, env, args.episodes,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        eval_every=args.eval_every, eval_opponents=_agents(args.eval_opponents),
        eval_deals=args.eval_deals, seed=args.seed, on_eval=show,
    )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        learner.save(out / "weights.json")
        write_history(history, out / "history.csv")
        print(f"Salvati {out / 'weights.json'} e {out / 'history.csv'}")


if __name__ == "__main__":
    main()
