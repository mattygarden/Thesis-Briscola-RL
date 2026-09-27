"""The memory experiment: basic vs card-counting agents, several training seeds.

    python -m briscola.experiment --episodes 100000 --seeds 3 --out results/memory

Each (condition, seed) pair trains a fresh :class:`LinearQLearner` under
identical settings; only the observation encoding differs. Runs execute in
parallel processes. Outputs: ``curves.csv`` (learning curves), ``final.csv``
(final evaluation per run and opponent), ``summary.csv`` and ``comparison.csv``
(Welch t-test between conditions, across seeds). To re-analyse saved results::

    python -m briscola.experiment --summarize results/memory
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .agents import AGENTS, make_agent
from .arena import evaluate
from .env import REWARD_MODES, BriscolaEnv
from .learners import LinearQLearner
from .rules import STANDARD, GameConfig
from .stats import holm_adjust, welch_t_test
from .train import EVAL_SEED, train

CONDITIONS = {"basic": False, "memory": True}


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int
    episodes: int
    opponents: tuple[str, ...]
    eval_opponents: tuple[str, ...]
    eval_every: int
    eval_deals: int
    final_deals: int
    reward: str
    lr: float
    reduced: bool
    out_dir: str | None


def run_one(spec: RunSpec) -> tuple[list[dict], list[dict]]:
    """Train and evaluate one agent; returns (curve rows, final rows)."""
    config = GameConfig.reduced() if spec.reduced else STANDARD
    memory = CONDITIONS[spec.condition]
    env = BriscolaEnv([make_agent(n) for n in spec.opponents], config,
                      memory=memory, reward=spec.reward)
    learner = LinearQLearner(config, memory, lr=spec.lr)
    eval_agents = [make_agent(n) for n in spec.eval_opponents]
    curve = train(learner, env, spec.episodes, eval_every=spec.eval_every,
                  eval_opponents=eval_agents, eval_deals=spec.eval_deals, seed=spec.seed)
    tag = {"condition": spec.condition, "seed": spec.seed}
    curve = [{**tag, **row} for row in curve]
    final = []
    for opp in eval_agents:
        # A different evaluation seed from the learning curve: fresh held-out deals.
        s = evaluate(learner, opp, spec.final_deals, config, seed=EVAL_SEED + 1)
        final.append({**tag, "opponent": opp.name, "games": s.games,
                      "mean_reward": s.mean_reward, "reward_ci95": s.reward_ci95,
                      "win_rate": s.wins / s.games, "mean_point_diff": s.mean_point_diff})
    if spec.out_dir:
        path = Path(spec.out_dir) / f"{spec.condition}-seed{spec.seed}.json"
        learner.save(path)
    return curve, final


def summarize(final: list[dict]) -> list[dict]:
    """Mean and spread across seeds of the final reward, per condition and opponent."""
    groups: dict[tuple[str, str], list[float]] = {}
    for row in final:
        groups.setdefault((row["condition"], row["opponent"]), []).append(float(row["mean_reward"]))
    out = []
    for (cond, opp), values in groups.items():
        n = len(values)
        mean = sum(values) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else float("nan")
        out.append({"condition": cond, "opponent": opp, "seeds": n,
                    "mean_reward": mean, "sd_across_seeds": sd})
    return sorted(out, key=lambda r: (r["opponent"], r["condition"]))


def compare_conditions(final: list[dict]) -> list[dict]:
    """Memory minus basic, per opponent, with each run (seed) as one observation.

    Per-opponent p-values are Holm-adjusted as one family. The extra
    "media" row compares the average over opponents (the training objective,
    since training uses the opponent mixture) and is not adjusted.
    """
    per_run: dict[tuple[str, int], dict[str, float]] = {}
    for row in final:
        per_run.setdefault((row["condition"], int(row["seed"])), {})[row["opponent"]] = float(row["mean_reward"])
    opponents = sorted({row["opponent"] for row in final})

    def values(cond: str, opp: str | None) -> list[float]:
        out = []
        for (c, _), by_opp in sorted(per_run.items()):
            if c == cond:
                out.append(by_opp[opp] if opp else sum(by_opp.values()) / len(by_opp))
        return out

    rows = []
    for opp in opponents + [None]:
        mem, base = values("memory", opp), values("basic", opp)
        if len(mem) < 2 or len(base) < 2:
            continue
        r = welch_t_test(mem, base)
        rows.append({"opponent": opp or "media", "basic": sum(base) / len(base),
                     "memory": sum(mem) / len(mem), "diff": r.diff, "t": r.t, "df": r.df,
                     "p_value": r.p_value, "p_holm": float("nan")})
    per_opp = [row for row in rows if row["opponent"] != "media"]
    for row, adj in zip(per_opp, holm_adjust([row["p_value"] for row in per_opp])):
        row["p_holm"] = adj
    return rows


def report(out: Path, final: list[dict]) -> None:
    """Write summary.csv and comparison.csv and print both."""
    summary = summarize(final)
    _write_csv(summary, out / "summary.csv")
    seeds = max(row["seeds"] for row in summary)
    print(f"\nReward medio finale per condizione, media su {seeds} seed:")
    for row in summary:
        print(f"  vs {row['opponent']:<7} {row['condition']:<7} "
              f"{row['mean_reward']:+.3f}  (sd tra seed {row['sd_across_seeds']:.3f})")
    comparison = compare_conditions(final)
    if comparison:
        _write_csv(comparison, out / "comparison.csv")
        print("\nMemoria meno base (test t di Welch tra seed; p di Holm sui singoli avversari):")
        for row in comparison:
            holm = "" if math.isnan(row["p_holm"]) else f"  p Holm {row['p_holm']:.3f}"
            print(f"  vs {row['opponent']:<7} diff {row['diff']:+.3f}  t {row['t']:+.2f}  "
                  f"gdl {row['df']:.1f}  p {row['p_value']:.3f}{holm}")
    print(f"Risultati in {out}/")


def _write_csv(rows: list[dict], path: Path) -> None:
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Esperimento memoria: agente base vs card counting.")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--opponents", default="random,lowest,greedy")
    parser.add_argument("--eval-opponents", default="random,lowest,greedy")
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-deals", type=int, default=300)
    parser.add_argument("--final-deals", type=int, default=2_000)
    parser.add_argument("--reward", default="trick_points", choices=REWARD_MODES)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--out", default="results/memory")
    parser.add_argument("--summarize", metavar="DIR", default=None,
                        help="rianalizza final.csv in DIR senza addestrare")
    args = parser.parse_args(argv)

    if args.summarize:
        with open(Path(args.summarize) / "final.csv", newline="") as f:
            report(Path(args.summarize), list(csv.DictReader(f)))
        return

    names = lambda s: tuple(n.strip() for n in s.split(",") if n.strip())  # noqa: E731
    for n in names(args.opponents) + names(args.eval_opponents):
        if n not in AGENTS:
            parser.error(f"unknown agent {n!r}; choose from {sorted(AGENTS)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        RunSpec(cond, seed, args.episodes, names(args.opponents), names(args.eval_opponents),
                args.eval_every, args.eval_deals, args.final_deals, args.reward, args.lr,
                args.reduced, str(out))
        for seed in range(args.seeds) for cond in CONDITIONS
    ]
    with open(out / "config.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(specs[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in specs)

    curves, finals = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for spec, (curve, final) in zip(specs, pool.map(run_one, specs)):
            curves += curve
            finals += final
            print(f"finito {spec.condition} seed {spec.seed}", flush=True)
    _write_csv(curves, out / "curves.csv")
    _write_csv(finals, out / "final.csv")
    report(out, finals)


if __name__ == "__main__":
    main()
