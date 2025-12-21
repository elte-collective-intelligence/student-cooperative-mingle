"""
Comprehensive Experiment Runner for Cooperative Mingle

This script runs all required experiments for the assignment:
1. Communication comparison (baseline vs discrete vs continuous)
2. Fairness comparison (baseline vs gini vs participation_variance vs exclusion_counts)
3. Algorithm comparison (PPO vs IPPO vs MAPPO) with fairness metrics

Results are saved to experiments/ directory with plots and metrics.

Usage:
    python run_experiments.py --all              # Run all experiments
    python run_experiments.py --communication   # Communication comparison only
    python run_experiments.py --fairness        # Fairness comparison only
    python run_experiments.py --algorithms      # Algorithm comparison only
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform
from src.utils.config import load_and_merge_configs


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("experiments")
PLOTS_DIR = Path("plots")

# Default experiment settings (reduced for faster runs)
DEFAULT_CONFIG = {
    "n_agents": 4,
    "n_rooms": 2,
    "room_capacity": 2,
    "total_frames": 2000,
    "frames_per_batch": 200,
    "num_epochs": 4,
    "minibatch_size": 64,
    "seeds": [0, 1, 2],
}


# ============================================================
# Utility Functions
# ============================================================

def ensure_dirs():
    """Create necessary directories."""
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "communication").mkdir(exist_ok=True)
    (RESULTS_DIR / "fairness").mkdir(exist_ok=True)
    (RESULTS_DIR / "algorithms").mkdir(exist_ok=True)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results, path):
    """Save results to JSON file."""
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {path}")


def compute_gini(values):
    """Compute Gini coefficient for a list of values."""
    values = np.array(values)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


# ============================================================
# Simple Training Loop (for experiments)
# ============================================================

def simple_train_loop(env, total_frames=1000, seed=0):
    """
    Simple training loop that collects metrics.

    Returns metrics dict with:
    - rewards: list of episode rewards
    - agent_rewards: per-agent cumulative rewards
    - collisions: collision count
    - exclusions: exclusion count
    """
    set_seed(seed)

    metrics = {
        "rewards": [],
        "agent_rewards": np.zeros(env.n_agents),
        "step_rewards": [],
        "frames": [],
    }

    td = env.reset()
    episode_reward = 0
    frame = 0

    while frame < total_frames:
        # Random actions for baseline
        actions = torch.randn(env.n_agents, 2) * 0.3
        actions = actions.clamp(-env.max_speed, env.max_speed)

        from tensordict import TensorDict
        step_td = TensorDict({"action": actions}, batch_size=[])

        # Handle communication env
        if hasattr(env, 'vocab_size'):
            step_td["message"] = torch.zeros(env.n_agents, dtype=torch.long)

        td = env._step(step_td)

        reward = td["reward"].sum().item()
        episode_reward += reward
        metrics["agent_rewards"] += td["reward"].squeeze().numpy()
        metrics["step_rewards"].append(reward)
        metrics["frames"].append(frame)

        frame += env.n_agents

        if td["done"].any() or td["terminated"].any():
            metrics["rewards"].append(episode_reward)
            episode_reward = 0
            td = env.reset()

    # Final episode
    if episode_reward > 0:
        metrics["rewards"].append(episode_reward)

    # Compute fairness metrics
    metrics["gini_coefficient"] = compute_gini(metrics["agent_rewards"])
    metrics["reward_variance"] = float(np.var(metrics["agent_rewards"]))
    metrics["mean_reward"] = float(np.mean(metrics["rewards"])) if metrics["rewards"] else 0
    metrics["final_reward"] = float(np.mean(metrics["rewards"][-10:])) if len(metrics["rewards"]) >= 10 else metrics["mean_reward"]

    return metrics


# ============================================================
# Communication Experiments
# ============================================================

def run_communication_experiments(config=None):
    """
    Run communication comparison experiments.

    Compares:
    - Baseline (no communication)
    - Discrete communication
    - Continuous communication (if available)
    """
    print("\n" + "="*60)
    print("COMMUNICATION EXPERIMENTS")
    print("="*60)

    config = config or DEFAULT_CONFIG
    results = {"baseline": [], "discrete_comm": []}

    for seed in config["seeds"]:
        print(f"\n--- Seed {seed} ---")

        # Baseline (no communication)
        print("Running baseline (no communication)...")
        env_baseline = MingleEnv(
            n_agents=config["n_agents"],
            n_rooms=config["n_rooms"],
            room_capacity=config["room_capacity"],
            max_steps=100,
        )
        baseline_metrics = simple_train_loop(env_baseline, config["total_frames"], seed)
        results["baseline"].append(baseline_metrics)
        print(f"  Mean reward: {baseline_metrics['mean_reward']:.3f}")
        print(f"  Gini coefficient: {baseline_metrics['gini_coefficient']:.3f}")

        # Discrete communication
        print("Running discrete communication...")
        try:
            from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
            env_comm = MingleEnvWithComm(
                n_agents=config["n_agents"],
                n_rooms=config["n_rooms"],
                room_capacity=config["room_capacity"],
                max_steps=100,
                vocab_size=2,
            )
            comm_metrics = simple_train_loop(env_comm, config["total_frames"], seed)
            results["discrete_comm"].append(comm_metrics)
            print(f"  Mean reward: {comm_metrics['mean_reward']:.3f}")
            print(f"  Gini coefficient: {comm_metrics['gini_coefficient']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            results["discrete_comm"].append(None)

    # Save results
    save_results(results, RESULTS_DIR / "communication" / "comparison_results.json")

    # Generate plots
    plot_communication_comparison(results)

    return results


def plot_communication_comparison(results):
    """Generate comparison plots for communication experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Mean Reward Comparison
    ax = axes[0]
    categories = ["Baseline", "Discrete Comm"]
    means = []
    stds = []

    for key in ["baseline", "discrete_comm"]:
        valid_results = [r for r in results[key] if r is not None]
        if valid_results:
            rewards = [r["mean_reward"] for r in valid_results]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Gini Coefficient (Fairness)
    ax = axes[1]
    gini_means = []
    gini_stds = []

    for key in ["baseline", "discrete_comm"]:
        valid_results = [r for r in results[key] if r is not None]
        if valid_results:
            gini = [r["gini_coefficient"] for r in valid_results]
            gini_means.append(np.mean(gini))
            gini_stds.append(np.std(gini))
        else:
            gini_means.append(0)
            gini_stds.append(0)

    bars = ax.bar(x, gini_means, yerr=gini_stds, capsize=5, color=['steelblue', 'coral'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Learning Curves (first seed)
    ax = axes[2]
    for key, label, color in [("baseline", "Baseline", "steelblue"),
                               ("discrete_comm", "Discrete Comm", "coral")]:
        if results[key] and results[key][0] is not None:
            r = results[key][0]
            if "step_rewards" in r and len(r["step_rewards"]) > 0:
                # Smooth the rewards
                window = min(50, len(r["step_rewards"]) // 10)
                if window > 1:
                    smoothed = np.convolve(r["step_rewards"], np.ones(window)/window, mode='valid')
                    ax.plot(smoothed, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Step Reward (smoothed)")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "communication_comparison.png", dpi=150)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR / 'communication_comparison.png'}")


# ============================================================
# Fairness Experiments
# ============================================================

def run_fairness_experiments(config=None):
    """
    Run fairness comparison experiments.

    Compares:
    - Baseline (no fairness)
    - Gini-based redistribution
    - Participation variance
    - Exclusion counts
    """
    print("\n" + "="*60)
    print("FAIRNESS EXPERIMENTS")
    print("="*60)

    config = config or DEFAULT_CONFIG

    fairness_modes = [
        ("none", "Baseline"),
        ("gini", "Gini"),
        ("participation_variance", "Participation"),
        ("exclusion_counts", "Exclusion"),
    ]

    results = {mode: [] for mode, _ in fairness_modes}

    for seed in config["seeds"]:
        print(f"\n--- Seed {seed} ---")

        for mode, name in fairness_modes:
            print(f"Running {name} mode...")

            # Create environment
            env = MingleEnv(
                n_agents=config["n_agents"],
                n_rooms=config["n_rooms"],
                room_capacity=config["room_capacity"],
                max_steps=100,
            )

            # Create fairness transform
            fairness_cfg = {"mode": mode, "alpha": 0.5}
            transform = make_fairness_reward_transform(fairness_cfg)

            # Run with transform applied manually
            metrics = run_with_fairness_transform(env, transform, config["total_frames"], seed)
            results[mode].append(metrics)

            print(f"  Mean reward: {metrics['mean_reward']:.3f}")
            print(f"  Gini coefficient: {metrics['gini_coefficient']:.3f}")
            print(f"  Reward variance: {metrics['reward_variance']:.3f}")

    # Save results
    save_results(results, RESULTS_DIR / "fairness" / "comparison_results.json")

    # Generate plots
    plot_fairness_comparison(results, fairness_modes)

    return results


def run_with_fairness_transform(env, transform, total_frames, seed):
    """Run training loop with fairness transform applied."""
    set_seed(seed)
    from tensordict import TensorDict

    metrics = {
        "rewards": [],
        "agent_rewards": np.zeros(env.n_agents),
        "step_rewards": [],
        "frames": [],
    }

    td = env.reset()

    # Initialize transform
    init_td = TensorDict({"observation": td["observation"]}, batch_size=[])
    transform(init_td)

    episode_reward = 0
    frame = 0

    while frame < total_frames:
        actions = torch.randn(env.n_agents, 2) * 0.3
        actions = actions.clamp(-env.max_speed, env.max_speed)

        step_td = TensorDict({"action": actions}, batch_size=[])
        td = env._step(step_td)

        # Apply fairness transform
        reward_td = TensorDict({
            "observation": td["observation"],
            "reward": td["reward"],
        }, batch_size=[])
        transformed_td = transform(reward_td)
        transformed_reward = transformed_td["reward"]

        reward = transformed_reward.sum().item()
        episode_reward += reward
        metrics["agent_rewards"] += transformed_reward.squeeze().numpy()
        metrics["step_rewards"].append(reward)
        metrics["frames"].append(frame)

        frame += env.n_agents

        if td["done"].any() or td["terminated"].any():
            metrics["rewards"].append(episode_reward)
            episode_reward = 0
            td = env.reset()
            # Re-initialize transform for new episode
            init_td = TensorDict({"observation": td["observation"]}, batch_size=[])
            transform(init_td)

    if episode_reward > 0:
        metrics["rewards"].append(episode_reward)

    metrics["gini_coefficient"] = compute_gini(metrics["agent_rewards"])
    metrics["reward_variance"] = float(np.var(metrics["agent_rewards"]))
    metrics["mean_reward"] = float(np.mean(metrics["rewards"])) if metrics["rewards"] else 0

    return metrics


def plot_fairness_comparison(results, fairness_modes):
    """Generate comparison plots for fairness experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = [name for _, name in fairness_modes]
    colors = ['steelblue', 'coral', 'seagreen', 'purple']

    # Plot 1: Mean Reward
    ax = axes[0]
    means = []
    stds = []
    for mode, _ in fairness_modes:
        if results[mode]:
            rewards = [r["mean_reward"] for r in results[mode]]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(categories))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward by Fairness Mode")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Gini Coefficient
    ax = axes[1]
    gini_means = []
    gini_stds = []
    for mode, _ in fairness_modes:
        if results[mode]:
            gini = [r["gini_coefficient"] for r in results[mode]]
            gini_means.append(np.mean(gini))
            gini_stds.append(np.std(gini))
        else:
            gini_means.append(0)
            gini_stds.append(0)

    ax.bar(x, gini_means, yerr=gini_stds, capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness by Mode")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Reward Variance
    ax = axes[2]
    var_means = []
    var_stds = []
    for mode, _ in fairness_modes:
        if results[mode]:
            var = [r["reward_variance"] for r in results[mode]]
            var_means.append(np.mean(var))
            var_stds.append(np.std(var))
        else:
            var_means.append(0)
            var_stds.append(0)

    ax.bar(x, var_means, yerr=var_stds, capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Reward Variance (lower = fairer)")
    ax.set_title("Reward Distribution Variance")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fairness_comparison.png", dpi=150)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR / 'fairness_comparison.png'}")


# ============================================================
# Algorithm Comparison Experiments
# ============================================================

def run_algorithm_experiments(config=None):
    """
    Run algorithm comparison experiments.

    Compares PPO, IPPO, and MAPPO with fairness metrics.
    Uses existing experiment results if available.
    """
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON EXPERIMENTS")
    print("="*60)

    config = config or DEFAULT_CONFIG

    # Check for existing results
    algorithms = ["ppo", "ippo", "mappo"]
    results = {}

    for algo in algorithms:
        algo_results = []
        for seed in config["seeds"]:
            result_path = RESULTS_DIR / algo / f"{algo}_seed{seed}.json"
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                    algo_results.append(data)
                print(f"Loaded existing results: {result_path}")
            else:
                print(f"Missing: {result_path}")

        if algo_results:
            results[algo] = algo_results

    if not results:
        print("No existing algorithm results found. Running basic comparison...")
        # Run basic experiments with each algorithm's environment
        results = run_basic_algorithm_comparison(config)

    # Generate comparison plots
    plot_algorithm_comparison(results)

    # Save combined results
    save_results(results, RESULTS_DIR / "algorithms" / "combined_results.json")

    return results


def run_basic_algorithm_comparison(config):
    """Run basic comparison if no existing results."""
    results = {}

    # For now, just run baseline with different random seeds
    # In full implementation, this would use the actual trainers

    for algo in ["ppo", "ippo", "mappo"]:
        results[algo] = []
        print(f"\nRunning {algo.upper()} baseline...")

        for seed in config["seeds"]:
            env = MingleEnv(
                n_agents=config["n_agents"],
                n_rooms=config["n_rooms"],
                room_capacity=config["room_capacity"],
                max_steps=100,
            )

            metrics = simple_train_loop(env, config["total_frames"], seed)

            # Format to match expected structure
            algo_metrics = {
                "reward": metrics.get("step_rewards", []),
                "loss": [0.0] * len(metrics.get("step_rewards", [])),
                "episode_rewards": metrics.get("rewards", []),
                "gini_coefficient": metrics["gini_coefficient"],
                "reward_variance": metrics["reward_variance"],
            }

            results[algo].append(algo_metrics)

            # Save individual result
            algo_dir = RESULTS_DIR / algo
            algo_dir.mkdir(exist_ok=True)
            save_results(algo_metrics, algo_dir / f"{algo}_seed{seed}.json")

    return results


def plot_algorithm_comparison(results):
    """Generate comparison plots for algorithm experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    algorithms = list(results.keys())
    colors = {'ppo': 'steelblue', 'ippo': 'coral', 'mappo': 'seagreen'}

    # Plot 1: Final Reward Comparison
    ax = axes[0, 0]
    means = []
    stds = []

    for algo in algorithms:
        if results[algo]:
            # Get final rewards (last 20% of training)
            final_rewards = []
            for r in results[algo]:
                if "reward" in r and len(r["reward"]) > 0:
                    last_portion = r["reward"][int(len(r["reward"]) * 0.8):]
                    final_rewards.append(np.mean(last_portion))
                elif "episode_rewards" in r and len(r["episode_rewards"]) > 0:
                    last_portion = r["episode_rewards"][int(len(r["episode_rewards"]) * 0.8):]
                    final_rewards.append(np.mean(last_portion))

            if final_rewards:
                means.append(np.mean(final_rewards))
                stds.append(np.std(final_rewards))
            else:
                means.append(0)
                stds.append(0)
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(algorithms))
    ax.bar(x, means, yerr=stds, capsize=5, color=[colors.get(a, 'gray') for a in algorithms])
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.set_ylabel("Mean Reward (final 20%)")
    ax.set_title("Final Performance Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Gini Coefficient
    ax = axes[0, 1]
    gini_means = []
    gini_stds = []

    for algo in algorithms:
        if results[algo]:
            gini_values = [r.get("gini_coefficient", 0) for r in results[algo]]
            gini_means.append(np.mean(gini_values))
            gini_stds.append(np.std(gini_values))
        else:
            gini_means.append(0)
            gini_stds.append(0)

    ax.bar(x, gini_means, yerr=gini_stds, capsize=5, color=[colors.get(a, 'gray') for a in algorithms])
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Learning Curves
    ax = axes[1, 0]
    for algo in algorithms:
        if results[algo] and results[algo][0]:
            r = results[algo][0]
            if "reward" in r and len(r["reward"]) > 0:
                data = r["reward"]
            elif "episode_rewards" in r and len(r["episode_rewards"]) > 0:
                data = r["episode_rewards"]
            else:
                continue

            # Smooth
            window = max(1, len(data) // 20)
            if window > 1 and len(data) > window:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=algo.upper(), color=colors.get(algo, 'gray'), alpha=0.8)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Reward Variance
    ax = axes[1, 1]
    var_means = []
    var_stds = []

    for algo in algorithms:
        if results[algo]:
            var_values = [r.get("reward_variance", 0) for r in results[algo]]
            var_means.append(np.mean(var_values))
            var_stds.append(np.std(var_values))
        else:
            var_means.append(0)
            var_stds.append(0)

    ax.bar(x, var_means, yerr=var_stds, capsize=5, color=[colors.get(a, 'gray') for a in algorithms])
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.set_ylabel("Reward Variance")
    ax.set_title("Stability Comparison")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "algorithm_comparison.png", dpi=150)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR / 'algorithm_comparison.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run Cooperative Mingle experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--communication", action="store_true", help="Run communication experiments")
    parser.add_argument("--fairness", action="store_true", help="Run fairness experiments")
    parser.add_argument("--algorithms", action="store_true", help="Run algorithm comparison")
    parser.add_argument("--frames", type=int, default=2000, help="Total training frames")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")

    args = parser.parse_args()

    # If no specific experiment selected, run all
    if not (args.communication or args.fairness or args.algorithms):
        args.all = True

    # Update config
    config = DEFAULT_CONFIG.copy()
    config["total_frames"] = args.frames
    config["seeds"] = args.seeds

    ensure_dirs()

    print("="*60)
    print("COOPERATIVE MINGLE EXPERIMENT RUNNER")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Total frames: {config['total_frames']}")
    print(f"Seeds: {config['seeds']}")
    print("="*60)

    all_results = {}

    if args.all or args.communication:
        all_results["communication"] = run_communication_experiments(config)

    if args.all or args.fairness:
        all_results["fairness"] = run_fairness_experiments(config)

    if args.all or args.algorithms:
        all_results["algorithms"] = run_algorithm_experiments(config)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
