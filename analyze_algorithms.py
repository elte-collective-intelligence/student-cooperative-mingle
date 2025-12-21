"""
Algorithm Comparison Analysis for Cooperative Mingle

This script analyzes and compares results across:
- PPO, IPPO, and MAPPO algorithms
- Includes fairness metrics (Gini coefficient, reward variance)
- Generates comparison plots and summary tables

Usage:
    python analyze_algorithms.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
EXP_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"

# Ensure plots directory exists
PLOTS_DIR.mkdir(exist_ok=True)

FILES = {
    "ppo": [
        EXP_DIR / "ppo" / "ppo_seed0.json",
        EXP_DIR / "ppo" / "ppo_seed1.json",
        EXP_DIR / "ppo" / "ppo_seed2.json",
    ],
    "ippo": [
        EXP_DIR / "ippo" / "ippo_seed0.json",
        EXP_DIR / "ippo" / "ippo_seed1.json",
        EXP_DIR / "ippo" / "ippo_seed2.json",
    ],
    "mappo": [
        EXP_DIR / "mappo" / "mappo_seed0.json",
        EXP_DIR / "mappo" / "mappo_seed1.json",
        EXP_DIR / "mappo" / "mappo_seed2.json",
    ],
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------- per-seed summaries ----------------------


def summarize_ppo_single(data):
    """PPO: use loss (no reward logged)."""
    loss = np.array(data["loss"])
    last = loss[int(len(loss) * 0.8):]
    return {
        "metric": "loss",
        "mean_last": float(last.mean()),
        "min": float(loss.min()),
    }


def summarize_ippo_single(data):
    rewards = np.array(data["reward"])
    last = rewards[int(len(rewards) * 0.8):]
    return {
        "metric": "reward",
        "mean_last": float(last.mean()),
        "max": float(rewards.max()),
    }


def summarize_mappo_single(data):
    rewards = np.array(data["episode_rewards"])
    last = rewards[int(len(rewards) * 0.8):]
    return {
        "metric": "episode_reward",
        "mean_last": float(last.mean()),
        "max": float(rewards.max()),
    }


# ---------------------- multi-seed aggregation ----------------------


def aggregate_stats(algo, per_seed_stats):
    metric = per_seed_stats[0]["metric"]
    means = np.array([s["mean_last"] for s in per_seed_stats])
    extrema = np.array(
        [s.get("max", s.get("min")) for s in per_seed_stats]
    )

    n = len(per_seed_stats)
    mean_mean = float(means.mean())
    std_mean = float(means.std(ddof=1)) if n > 1 else 0.0
    sem_mean = std_mean / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem_mean if n > 1 else 0.0

    mean_ext = float(extrema.mean())

    return {
        "metric": metric,
        "n_seeds": n,
        "mean_last_mean": mean_mean,
        "mean_last_std": std_mean,
        "mean_last_ci95": ci95,
        "max_or_min_mean": mean_ext,
    }


def compute_gini(values):
    """Compute Gini coefficient for a list of values."""
    values = np.array(values)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def extract_fairness_metrics(data):
    """Extract fairness metrics from experiment data."""
    metrics = {
        "gini_coefficient": data.get("gini_coefficient", 0.0),
        "reward_variance": data.get("reward_variance", 0.0),
    }

    # If we have per-step rewards, compute additional metrics
    if "reward" in data and len(data["reward"]) > 0:
        rewards = np.array(data["reward"])
        metrics["reward_std"] = float(np.std(rewards))
        metrics["reward_range"] = float(np.max(rewards) - np.min(rewards))

    return metrics


def generate_comparison_table(summaries):
    """Generate a formatted comparison table."""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)

    # Header
    print(f"{'Algorithm':<10} | {'Metric':<15} | {'Seeds':<6} | {'Mean (last 20%)':<18} | {'95% CI':<10} | {'Best':<10}")
    print("-"*80)

    for algo, s in summaries.items():
        label = "Max" if "reward" in s["metric"] else "Min"
        print(
            f"{algo.upper():<10} | {s['metric']:<15} | {s['n_seeds']:<6} | "
            f"{s['mean_last_mean']:>18.3f} | +/- {s['mean_last_ci95']:<6.3f} | "
            f"{label}: {s['max_or_min_mean']:.3f}"
        )

    print("="*80)


def generate_fairness_table(fairness_data):
    """Generate fairness comparison table."""
    print("\n" + "="*80)
    print("FAIRNESS METRICS COMPARISON")
    print("="*80)

    print(f"{'Algorithm':<10} | {'Gini Coef.':<12} | {'Reward Var.':<12} | {'Reward Std':<12}")
    print("-"*80)

    for algo, metrics in fairness_data.items():
        print(
            f"{algo.upper():<10} | {metrics['gini']:.4f}       | "
            f"{metrics['variance']:.4f}       | {metrics['std']:.4f}"
        )

    print("="*80)


def plot_comprehensive_comparison(summaries, fairness_data):
    """Generate comprehensive comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    algorithms = list(summaries.keys())
    colors = {'ppo': 'steelblue', 'ippo': 'coral', 'mappo': 'seagreen'}

    # Plot 1: Performance (Mean Reward/Loss)
    ax = axes[0, 0]
    means = [summaries[a]['mean_last_mean'] for a in algorithms]
    cis = [summaries[a]['mean_last_ci95'] for a in algorithms]
    x = np.arange(len(algorithms))
    bars = ax.bar(x, means, yerr=cis, capsize=5,
                  color=[colors.get(a, 'gray') for a in algorithms])
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.set_ylabel("Performance (last 20%)")
    ax.set_title("Algorithm Performance Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.01,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Gini Coefficient (Fairness)
    ax = axes[0, 1]
    if fairness_data:
        gini_vals = [fairness_data.get(a, {}).get('gini', 0) for a in algorithms]
        bars = ax.bar(x, gini_vals, color=[colors.get(a, 'gray') for a in algorithms])
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algorithms])
        ax.set_ylabel("Gini Coefficient (lower = fairer)")
        ax.set_title("Fairness Comparison (Gini)")
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

    # Plot 3: Reward Variance (Stability)
    ax = axes[1, 0]
    if fairness_data:
        var_vals = [fairness_data.get(a, {}).get('variance', 0) for a in algorithms]
        bars = ax.bar(x, var_vals, color=[colors.get(a, 'gray') for a in algorithms])
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algorithms])
        ax.set_ylabel("Reward Variance")
        ax.set_title("Stability Comparison")
        ax.grid(axis='y', alpha=0.3)

    # Plot 4: Learning Curves (all algorithms)
    ax = axes[1, 1]
    for algo in algorithms:
        paths = FILES.get(algo, [])
        valid_paths = [p for p in paths if p.exists()]
        if valid_paths:
            # Plot first seed
            data = load_json(valid_paths[0])
            if "reward" in data and len(data["reward"]) > 0:
                rewards = np.array(data["reward"])
                # Smooth
                window = max(1, len(rewards) // 20)
                if window > 1 and len(rewards) > window:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(smoothed, label=algo.upper(), color=colors.get(algo, 'gray'), alpha=0.8)
            elif "loss" in data and len(data["loss"]) > 0:
                loss = np.array(data["loss"])
                window = max(1, len(loss) // 20)
                if window > 1 and len(loss) > window:
                    smoothed = np.convolve(loss, np.ones(window)/window, mode='valid')
                    ax.plot(smoothed, label=f"{algo.upper()} (loss)",
                           color=colors.get(algo, 'gray'), alpha=0.8, linestyle='--')

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Metric Value")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "algorithm_comparison_comprehensive.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'algorithm_comparison_comprehensive.png'}")


def main():
    summaries = {}
    fairness_data = {}

    for algo, paths in FILES.items():
        per_seed_stats = []
        all_gini = []
        all_variance = []
        all_std = []

        for p in paths:
            if not p.exists():
                continue

            data = load_json(p)

            if algo == "ppo":
                s = summarize_ppo_single(data)
            elif algo == "ippo":
                s = summarize_ippo_single(data)
            elif algo == "mappo":
                s = summarize_mappo_single(data)
            else:
                continue

            per_seed_stats.append(s)

            # Extract fairness metrics
            fm = extract_fairness_metrics(data)
            all_gini.append(fm["gini_coefficient"])
            all_variance.append(fm["reward_variance"])
            if "reward_std" in fm:
                all_std.append(fm["reward_std"])

        if not per_seed_stats:
            continue

        summaries[algo] = aggregate_stats(algo, per_seed_stats)

        # Aggregate fairness metrics
        fairness_data[algo] = {
            "gini": np.mean(all_gini) if all_gini else 0.0,
            "variance": np.mean(all_variance) if all_variance else 0.0,
            "std": np.mean(all_std) if all_std else 0.0,
        }

    # Generate tables
    generate_comparison_table(summaries)
    generate_fairness_table(fairness_data)

    # Generate comprehensive plots
    plot_comprehensive_comparison(summaries, fairness_data)

    # Individual algorithm plots
    for algo in ["ppo", "ippo", "mappo"]:
        paths = [p for p in FILES.get(algo, []) if p.exists()]
        if paths:
            plt.figure(figsize=(10, 6))
            for i, p in enumerate(paths):
                data = load_json(p)
                if "reward" in data and len(data["reward"]) > 0:
                    rewards = np.array(data["reward"])
                    plt.plot(rewards, alpha=0.6, label=f"Seed {i}")
                elif "loss" in data and len(data["loss"]) > 0:
                    loss = np.array(data["loss"])
                    plt.plot(loss, alpha=0.6, label=f"Seed {i}")

            plt.xlabel("Update Index")
            plt.ylabel("Metric Value")
            plt.title(f"{algo.upper()} - Multi-seed Comparison")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(PLOTS_DIR / f"{algo}_multi_seed.png", dpi=150)
            plt.close()
            print(f"Saved: {PLOTS_DIR / f'{algo}_multi_seed.png'}")

    # Save summary to JSON
    summary_json = {
        "summaries": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                         for kk, vv in v.items()}
                     for k, v in summaries.items()},
        "fairness": fairness_data,
    }
    with open(PLOTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved: {PLOTS_DIR / 'analysis_summary.json'}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
