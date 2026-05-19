"""
Pareto analysis for multi-objective sweeps.

Usage:
  python analyze_pareto.py --sweeps outputs/sweeps --out contribution_tests_and_comparisions/pareto
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(metrics: Dict, last_n: int = 20) -> Dict:
    rewards = metrics.get("episode_rewards", [])
    gini = metrics.get("gini_coefficients", [])
    fairness = metrics.get("episode_fairness", [])
    efficiency = metrics.get("episode_efficiency", [])

    def tail_mean(values: List[float]) -> float:
        if not values:
            return 0.0
        n = min(last_n, len(values))
        return float(np.mean(values[-n:]))

    gini_mean = tail_mean(gini)
    fairness_mean = tail_mean(fairness) if fairness else max(0.0, 1.0 - gini_mean)

    return {
        "reward_mean": tail_mean(rewards),
        "gini_mean": gini_mean,
        "fairness_mean": fairness_mean,
        "efficiency_mean": tail_mean(efficiency),
    }


def is_dominated(p: Tuple[float, float], others: List[Tuple[float, float]]) -> bool:
    for q in others:
        if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
            return True
    return False


def pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    frontier = []
    for p in points:
        if not is_dominated(p, points):
            frontier.append(p)
    return sorted(frontier, key=lambda x: x[0])


def group_key(cfg: Dict) -> Tuple:
    return (
        float(cfg.get("fairness", {}).get("alpha", 0.0)),
        int(cfg.get("env", {}).get("n_agents", 0)),
        int(cfg.get("env", {}).get("n_rooms", 0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=str, default="outputs/sweeps")
    parser.add_argument("--out", type=str, default="contribution_tests_and_comparisions/pareto")
    parser.add_argument("--last-n", type=int, default=20)
    args = parser.parse_args()

    sweeps_dir = Path(args.sweeps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = list(sweeps_dir.glob("**/metrics.json"))
    if not metrics_files:
        print(f"No metrics.json found under {sweeps_dir}")
        return 1

    rows = []
    for path in metrics_files:
        metrics = load_json(path)
        cfg = metrics.get("config", {})
        summary = summarize_run(metrics, last_n=args.last_n)

        row = {
            "path": str(path),
            "alpha": float(cfg.get("fairness", {}).get("alpha", 0.0)),
            "seed": int(cfg.get("seed", 0)),
            "n_agents": int(cfg.get("env", {}).get("n_agents", 0)),
            "n_rooms": int(cfg.get("env", {}).get("n_rooms", 0)),
            "reward_mean": summary["reward_mean"],
            "gini_mean": summary["gini_mean"],
            "fairness_mean": summary["fairness_mean"],
            "efficiency_mean": summary["efficiency_mean"],
        }
        rows.append(row)

    # Aggregate across seeds for each (alpha, n_agents, n_rooms)
    grouped: Dict[Tuple, List[Dict]] = {}
    for r in rows:
        key = (r["alpha"], r["n_agents"], r["n_rooms"])
        grouped.setdefault(key, []).append(r)

    aggregated = []
    for key, items in grouped.items():
        alpha, n_agents, n_rooms = key
        def mean_field(name: str) -> float:
            return float(np.mean([i[name] for i in items]))

        aggregated.append({
            "alpha": alpha,
            "n_agents": n_agents,
            "n_rooms": n_rooms,
            "reward_mean": mean_field("reward_mean"),
            "gini_mean": mean_field("gini_mean"),
            "fairness_mean": mean_field("fairness_mean"),
            "efficiency_mean": mean_field("efficiency_mean"),
            "n_seeds": len(items),
        })

    # Write aggregated CSV
    csv_path = out_dir / "pareto_aggregated.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = [
            "alpha",
            "n_agents",
            "n_rooms",
            "reward_mean",
            "gini_mean",
            "fairness_mean",
            "efficiency_mean",
            "n_seeds",
        ]
        f.write(",".join(header) + "\n")
        for r in aggregated:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # Split regimes
    small = [r for r in aggregated if r["n_agents"] in (4, 6) and r["n_rooms"] in (2, 3)]
    large = [r for r in aggregated if r["n_agents"] in (8, 10) and r["n_rooms"] in (4, 5)]

    def plot_scatter(data: List[Dict], title: str, out_name: str) -> None:
        if not data:
            return
        alphas = sorted(set(r["alpha"] for r in data))
        cmap = plt.get_cmap("viridis", len(alphas))
        alpha_to_color = {a: cmap(i) for i, a in enumerate(alphas)}

        fig, ax = plt.subplots(figsize=(7, 6))
        for r in data:
            ax.scatter(
                r["efficiency_mean"],
                r["fairness_mean"],
                color=alpha_to_color[r["alpha"]],
                alpha=0.8,
                label=f"alpha={r['alpha']:.2f}",
            )

        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)

        points = [(r["efficiency_mean"], r["fairness_mean"]) for r in data]
        frontier = pareto_frontier(points)
        if frontier:
            xs, ys = zip(*frontier)
            ax.plot(xs, ys, color="black", linewidth=2, label="Pareto frontier")

        ax.set_xlabel("Efficiency (mean)")
        ax.set_ylabel("Fairness (mean)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=200)
        plt.close(fig)

    plot_scatter(small, "Pareto Frontier (Small Scenarios)", "pareto_small.png")
    plot_scatter(large, "Pareto Frontier (Large Scenarios)", "pareto_large.png")

    # Combined scatter (all regimes)
    plot_scatter(aggregated, "Pareto Frontier (All Scenarios)", "pareto_all.png")


    # Additional simple plots for alpha trends across all aggregated runs
    if aggregated:
        alpha_values = sorted(set(r["alpha"] for r in aggregated))
        alpha_reward = []
        alpha_gini = []
        for alpha in alpha_values:
            subset = [r for r in aggregated if r["alpha"] == alpha]
            alpha_reward.append(float(np.mean([r["reward_mean"] for r in subset])))
            alpha_gini.append(float(np.mean([r["gini_mean"] for r in subset])))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(alpha_values, alpha_reward, marker="o")
        ax.set_xlabel("Alpha: efficiency weight")
        ax.set_ylabel("Mean reward")
        ax.set_title("Alpha vs Mean Reward")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "alpha_vs_reward.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(alpha_values, alpha_gini, marker="o")
        ax.set_xlabel("Alpha: efficiency weight")
        ax.set_ylabel("Mean Gini coefficient")
        ax.set_title("Alpha vs Gini")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "alpha_vs_gini.png", dpi=200)
        plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
