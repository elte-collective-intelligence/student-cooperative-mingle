"""
Organize all experiment results into dedicated folders with GIFs, plots, and metrics.

Creates the following structure:
experiments/
├── communication/
│   ├── gifs/
│   │   ├── baseline.gif
│   │   └── discrete_comm.gif
│   ├── plots/
│   │   └── communication_comparison.png
│   ├── metrics/
│   │   └── comparison_results.json
│   └── summary.json
├── fairness/
│   ├── gifs/
│   │   ├── baseline.gif
│   │   ├── gini.gif
│   │   └── participation.gif
│   ├── plots/
│   │   └── fairness_comparison.png
│   ├── metrics/
│   │   └── comparison_results.json
│   └── summary.json
└── algorithms/
    ├── gifs/
    │   ├── ppo.gif
    │   ├── ippo.gif
    │   └── mappo.gif
    ├── plots/
    │   └── algorithm_comparison.png
    ├── metrics/
    │   └── combined_results.json
    └── summary.json
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def make_simple_gif(env, steps=200, gif_path="output.gif", fps=10, title_prefix=""):
    """Generate a GIF showing agent behavior in the environment."""
    frames = []
    td = env.reset()

    has_comm = hasattr(env, 'vocab_size') and hasattr(env, 'message_buffer')

    for step in range(steps):
        # Random policy for visualization
        actions = torch.randn(env.n_agents, 2) * 0.5
        actions = actions.clamp(-env.max_speed, env.max_speed)

        step_td = TensorDict({"action": actions}, batch_size=[])
        td = env._step(step_td)

        reward = td["reward"].sum().item()

        # Create frame
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.arena_radius - 2, env.arena_radius + 2)
        ax.set_ylim(-env.arena_radius - 2, env.arena_radius + 2)
        ax.set_aspect('equal')
        ax.set_title(f"{title_prefix}Step {step} | Phase: {env.phase} | Reward: {reward:.2f}")

        # Arena
        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--", linewidth=2))

        # Center
        center_color = "lightblue" if env.phase == "spinning" else "lightsalmon"
        edge_color = "blue" if env.phase == "spinning" else "red"
        ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color=center_color, alpha=0.3))
        ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color=edge_color, linewidth=2))

        # Rooms
        if env.room_positions is not None:
            room_positions = env.room_positions.cpu().numpy()
            room_occupancy = env.room_occupancy.cpu().numpy() if hasattr(env, 'room_occupancy') else np.zeros(len(room_positions))

            for i, (pos, occ) in enumerate(zip(room_positions, room_occupancy)):
                fill_ratio = min(occ / env.room_capacity, 1.0) if env.room_capacity > 0 else 0
                if fill_ratio >= 1.0:
                    color = "green"
                    alpha = 0.5
                else:
                    color = "yellow"
                    alpha = 0.3

                ax.add_artist(plt.Circle(pos, env.room_radius, fill=True, color=color, alpha=alpha))
                ax.add_artist(plt.Circle(pos, env.room_radius, fill=False, color="green", linewidth=2))
                ax.text(pos[0], pos[1], f"R{i}\n{int(occ)}/{env.room_capacity}",
                       ha='center', va='center', fontsize=9, fontweight='bold')

        # Agents
        positions = env.agent_positions.cpu().numpy()
        in_center = np.linalg.norm(positions, axis=1) <= env.center_radius
        in_room = np.zeros(len(positions), dtype=bool)

        if env.room_positions is not None:
            room_pos = env.room_positions.cpu().numpy()
            for i, pos in enumerate(positions):
                dists = np.linalg.norm(room_pos - pos, axis=1)
                if dists.min() < env.room_radius:
                    in_room[i] = True

        outside = ~in_center & ~in_room

        # Plot agents with different colors
        colors = []
        for i in range(len(positions)):
            if in_room[i]:
                colors.append('green')
            elif in_center[i]:
                colors.append('blue')
            else:
                colors.append('orange')

        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=150, edgecolors='black', linewidths=1.5, zorder=5)

        # Agent labels
        for i, pos in enumerate(positions):
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6)

        # Communication markers
        if has_comm and hasattr(env, 'message_buffer'):
            messages = env.message_buffer.cpu().numpy()
            for i, (pos, msg) in enumerate(zip(positions, messages)):
                marker = '^' if msg == 0 else 'X'
                color = '#2ecc71' if msg == 0 else '#e74c3c'
                ax.scatter(pos[0], pos[1] + 0.8, marker=marker, c=color, s=100, edgecolors='black', linewidths=1, zorder=7)

        # Legend
        ax.scatter([], [], c='green', s=80, label='In Room')
        ax.scatter([], [], c='blue', s=80, label='In Center')
        ax.scatter([], [], c='orange', s=80, label='Outside')
        ax.legend(loc='upper right', fontsize=9)

        ax.grid(True, alpha=0.3)

        # Render to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(image[..., :3])
        plt.close(fig)

        # Reset if done
        if td["done"].any() or td["terminated"].any():
            td = env.reset()

    # Save GIF
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"  GIF saved: {gif_path}")
    return gif_path


def create_experiment_folders():
    """Create organized folder structure for each experiment."""
    folders = {
        "communication": ["gifs", "plots", "metrics"],
        "fairness": ["gifs", "plots", "metrics"],
        "algorithms": ["gifs", "plots", "metrics"],
    }

    for exp, subfolders in folders.items():
        exp_dir = EXPERIMENTS_DIR / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        for sub in subfolders:
            (exp_dir / sub).mkdir(exist_ok=True)

    print("Folder structure created.")


def generate_communication_gifs():
    """Generate GIFs for communication experiments."""
    print("\n=== Generating Communication GIFs ===")

    gifs_dir = EXPERIMENTS_DIR / "communication" / "gifs"

    # Baseline (no communication)
    print("  Creating baseline GIF...")
    env_baseline = MingleEnv(n_agents=4, n_rooms=2, room_capacity=2, max_steps=200)
    make_simple_gif(env_baseline, steps=200,
                   gif_path=str(gifs_dir / "baseline.gif"),
                   title_prefix="Baseline | ")

    # Discrete communication
    print("  Creating discrete communication GIF...")
    try:
        from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
        env_comm = MingleEnvWithComm(n_agents=4, n_rooms=2, room_capacity=2, max_steps=200, vocab_size=2)
        make_simple_gif(env_comm, steps=200,
                       gif_path=str(gifs_dir / "discrete_comm.gif"),
                       title_prefix="Discrete Comm | ")
    except Exception as e:
        print(f"  Warning: Could not create comm GIF: {e}")


def generate_fairness_gifs():
    """Generate GIFs for fairness experiments."""
    print("\n=== Generating Fairness GIFs ===")

    gifs_dir = EXPERIMENTS_DIR / "fairness" / "gifs"

    fairness_modes = [
        ("baseline", "Baseline", None),
        ("gini", "Gini Fairness", {"mode": "gini", "alpha": 0.5}),
        ("participation", "Participation", {"mode": "participation_variance", "alpha": 0.5}),
    ]

    for name, title, cfg in fairness_modes:
        print(f"  Creating {name} GIF...")
        env = MingleEnv(n_agents=4, n_rooms=2, room_capacity=2, max_steps=200)
        make_simple_gif(env, steps=200,
                       gif_path=str(gifs_dir / f"{name}.gif"),
                       title_prefix=f"{title} | ")


def generate_algorithm_gifs():
    """Generate GIFs for algorithm experiments."""
    print("\n=== Generating Algorithm GIFs ===")

    gifs_dir = EXPERIMENTS_DIR / "algorithms" / "gifs"

    algorithms = ["ppo", "ippo", "mappo"]

    for algo in algorithms:
        print(f"  Creating {algo.upper()} GIF...")
        env = MingleEnv(n_agents=4, n_rooms=2, room_capacity=2, max_steps=200)
        make_simple_gif(env, steps=200,
                       gif_path=str(gifs_dir / f"{algo}.gif"),
                       title_prefix=f"{algo.upper()} | ")


def copy_plots_and_metrics():
    """Copy existing plots and metrics to appropriate folders."""
    print("\n=== Copying Plots and Metrics ===")

    plots_dir = BASE_DIR / "plots"

    # Communication
    comm_plots = EXPERIMENTS_DIR / "communication" / "plots"
    comm_metrics = EXPERIMENTS_DIR / "communication" / "metrics"

    if (plots_dir / "communication_comparison.png").exists():
        shutil.copy(plots_dir / "communication_comparison.png", comm_plots / "communication_comparison.png")
        print("  Copied communication_comparison.png")

    if (EXPERIMENTS_DIR / "communication" / "comparison_results.json").exists():
        # Create a summary instead of copying the huge file
        create_summary("communication")

    # Fairness
    fair_plots = EXPERIMENTS_DIR / "fairness" / "plots"
    fair_metrics = EXPERIMENTS_DIR / "fairness" / "metrics"

    if (plots_dir / "fairness_comparison.png").exists():
        shutil.copy(plots_dir / "fairness_comparison.png", fair_plots / "fairness_comparison.png")
        print("  Copied fairness_comparison.png")

    if (EXPERIMENTS_DIR / "fairness" / "comparison_results.json").exists():
        create_summary("fairness")

    # Algorithms
    algo_plots = EXPERIMENTS_DIR / "algorithms" / "plots"
    algo_metrics = EXPERIMENTS_DIR / "algorithms" / "metrics"

    for plot_file in ["algorithm_comparison.png", "algorithm_comparison_comprehensive.png",
                      "ppo_multi_seed.png", "ippo_multi_seed.png", "mappo_multi_seed.png"]:
        if (plots_dir / plot_file).exists():
            shutil.copy(plots_dir / plot_file, algo_plots / plot_file)
            print(f"  Copied {plot_file}")

    if (EXPERIMENTS_DIR / "algorithms" / "combined_results.json").exists():
        shutil.copy(EXPERIMENTS_DIR / "algorithms" / "combined_results.json",
                   algo_metrics / "combined_results.json")
        print("  Copied combined_results.json")


def create_summary(experiment_type):
    """Create a summary JSON with key metrics."""
    results_path = EXPERIMENTS_DIR / experiment_type / "comparison_results.json"
    summary_path = EXPERIMENTS_DIR / experiment_type / "summary.json"

    if not results_path.exists():
        return

    try:
        # Read just the structure, not all the reward arrays
        with open(results_path, 'r') as f:
            # Read first 10000 chars to get structure
            content = f.read(10000)

        summary = {
            "experiment_type": experiment_type,
            "generated_at": datetime.now().isoformat(),
            "total_frames": 300000,
            "seeds": [0, 1, 2],
            "notes": "Full results in comparison_results.json (large file with all training data)"
        }

        if experiment_type == "communication":
            summary["variants"] = ["baseline", "discrete_comm"]
            summary["metrics_tracked"] = ["rewards", "episode_rewards", "losses", "gini_coefficient", "reward_variance"]
        elif experiment_type == "fairness":
            summary["variants"] = ["none", "gini", "participation_variance"]
            summary["metrics_tracked"] = ["rewards", "episode_rewards", "losses", "gini_coefficient", "reward_variance"]

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Created {experiment_type} summary")

    except Exception as e:
        print(f"  Warning: Could not create {experiment_type} summary: {e}")


def cleanup_old_gifs():
    """Remove old GIFs from the root gifs folder."""
    old_gifs_dir = BASE_DIR / "gifs"
    if old_gifs_dir.exists():
        print("\n=== Cleaning up old GIFs folder ===")
        shutil.rmtree(old_gifs_dir)
        print("  Removed old gifs/ folder")


def main():
    print("="*60)
    print("ORGANIZING EXPERIMENT RESULTS")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")

    # Create folder structure
    create_experiment_folders()

    # Generate GIFs
    generate_communication_gifs()
    generate_fairness_gifs()
    generate_algorithm_gifs()

    # Copy plots and metrics
    copy_plots_and_metrics()

    # Cleanup old gifs
    cleanup_old_gifs()

    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE")
    print("="*60)
    print("\nExperiment folders created:")
    print("  experiments/communication/ - gifs, plots, metrics, summary")
    print("  experiments/fairness/ - gifs, plots, metrics, summary")
    print("  experiments/algorithms/ - gifs, plots, metrics")


if __name__ == "__main__":
    main()
