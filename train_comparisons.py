"""
Comprehensive Training Script for Communication and Fairness Comparisons

This script trains real PPO policies for:
1. Communication comparison: baseline vs discrete communication
2. Fairness comparison: baseline vs gini vs participation_variance

Results are saved to experiments/ with proper metrics for comparison plots.

Usage:
    python train_comparisons.py --all
    python train_comparisons.py --communication
    python train_comparisons.py --fairness
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
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform
from src.models.policy_factory import build_policy
from src.models.critic_factory import build_critic


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("experiments")
PLOTS_DIR = Path("plots")

DEFAULT_CONFIG = {
    "n_agents": 4,
    "n_rooms": 2,
    "room_capacity": 2,
    "total_frames": 300000,  # 300k frames for proper training
    "frames_per_batch": 1000,
    "num_epochs": 4,
    "minibatch_size": 128,
    "lr": 3e-4,
    "gamma": 0.99,
    "lmbda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "seeds": [0, 1, 2],
}


def ensure_dirs():
    """Create output directories."""
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "communication").mkdir(exist_ok=True)
    (RESULTS_DIR / "fairness").mkdir(exist_ok=True)


def set_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gini(values):
    """Compute Gini coefficient."""
    values = np.array(values).flatten()
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.abs(values)  # Handle negative values
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


# ============================================================
# Simple MLP Policy and Critic
# ============================================================

class SimpleMLP(nn.Module):
    """Simple MLP for policy/critic."""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Handle multi-agent observations
        if x.dim() == 3:  # (batch, n_agents, obs_dim)
            batch_size, n_agents, obs_dim = x.shape
            x = x.view(-1, obs_dim)
            out = self.net(x)
            return out.view(batch_size, n_agents, -1)
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Policy network outputting mean and std for continuous actions."""
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.net = SimpleMLP(obs_dim, action_dim * 2, hidden_dim)
        self.min_std = 0.1

    def forward(self, obs):
        out = self.net(obs)
        # Split into loc and scale
        if out.dim() == 3:
            loc = out[..., :out.shape[-1]//2]
            scale = torch.nn.functional.softplus(out[..., out.shape[-1]//2:]) + self.min_std
        else:
            loc = out[..., :out.shape[-1]//2]
            scale = torch.nn.functional.softplus(out[..., out.shape[-1]//2:]) + self.min_std
        return loc, scale


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""
    def __init__(self, obs_dim, n_agents, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.net = SimpleMLP(obs_dim, 1, hidden_dim)

    def forward(self, obs):
        return self.net(obs)


# ============================================================
# PPO Trainer
# ============================================================

class SimplePPOTrainer:
    """Simple PPO trainer for comparison experiments."""

    def __init__(self, env, config, device, fairness_transform=None):
        self.env = env
        self.config = config
        self.device = device
        self.fairness_transform = fairness_transform

        # Get dimensions
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_dim = 2  # 2D movement
        n_agents = env.n_agents

        # Create networks
        self.policy = PolicyNetwork(obs_dim, action_dim, n_agents).to(device)
        self.critic = CriticNetwork(obs_dim, n_agents).to(device)

        # Optimizers
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=config["lr"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config["lr"])

        # Metrics
        self.metrics = {
            "rewards": [],
            "episode_rewards": [],
            "losses": [],
            "policy_losses": [],
            "value_losses": [],
            "agent_rewards": [],
            "gini_coefficients": [],
        }

    def collect_rollout(self, num_steps):
        """Collect experience from environment."""
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        td = self.env.reset()
        obs = td["observation"].to(self.device)

        episode_reward = 0
        agent_episode_rewards = torch.zeros(self.env.n_agents)

        for _ in range(num_steps):
            with torch.no_grad():
                # Get policy output
                loc, scale = self.policy(obs.unsqueeze(0))
                loc = loc.squeeze(0)
                scale = scale.squeeze(0)

                # Sample action
                dist = torch.distributions.Normal(loc, scale)
                action = dist.sample()
                action = action.clamp(-self.env.max_speed, self.env.max_speed)
                log_prob = dist.log_prob(action).sum(dim=-1)

                # Get value
                value = self.critic(obs.unsqueeze(0)).squeeze(0)

            # Step environment
            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            next_td = self.env._step(step_td)

            reward = next_td["reward"].to(self.device)

            # Apply fairness transform if provided
            if self.fairness_transform is not None:
                transform_td = TensorDict({
                    "observation": next_td["observation"],
                    "reward": reward.cpu(),
                }, batch_size=[])
                transformed = self.fairness_transform(transform_td)
                reward = transformed["reward"].to(self.device)

            done = next_td["done"].any() or next_td["terminated"].any()

            # Store experience
            observations.append(obs)
            actions.append(action)
            rewards.append(reward.squeeze())
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.squeeze())

            # Track rewards
            episode_reward += reward.sum().item()
            agent_episode_rewards += reward.squeeze().cpu()

            # Track per-step reward
            self.metrics["rewards"].append(reward.mean().item())

            if done:
                self.metrics["episode_rewards"].append(episode_reward)
                self.metrics["agent_rewards"].append(agent_episode_rewards.numpy().copy())
                self.metrics["gini_coefficients"].append(compute_gini(agent_episode_rewards.numpy()))

                episode_reward = 0
                agent_episode_rewards = torch.zeros(self.env.n_agents)
                td = self.env.reset()
                obs = td["observation"].to(self.device)

                # Re-init fairness transform
                if self.fairness_transform is not None:
                    init_td = TensorDict({"observation": obs.cpu()}, batch_size=[])
                    self.fairness_transform(init_td)
            else:
                obs = next_td["observation"].to(self.device)

        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "rewards": torch.stack(rewards),
            "dones": torch.tensor(dones, device=self.device),
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
        }

    def compute_gae(self, rewards, values, dones, gamma=0.99, lmbda=0.95):
        """Compute Generalized Advantage Estimation for multi-agent setting."""
        # rewards: (T, n_agents), values: (T, n_agents), dones: (T,)
        T, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(n_agents, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(n_agents, device=rewards.device)
            else:
                next_value = values[t + 1]

            done_mask = (1 - dones[t].float())
            delta = rewards[t] + gamma * next_value * done_mask - values[t]
            advantages[t] = delta + gamma * lmbda * done_mask * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """Update policy and critic."""
        observations = rollout["observations"]  # (T, n_agents, obs_dim)
        actions = rollout["actions"]  # (T, n_agents, 2)
        old_log_probs = rollout["log_probs"]  # (T, n_agents)
        rewards = rollout["rewards"]  # (T, n_agents)
        values = rollout["values"]  # (T, n_agents)
        dones = rollout["dones"]  # (T,)

        # Compute advantages per agent
        advantages, returns = self.compute_gae(
            rewards, values, dones,
            self.config["gamma"], self.config["lmbda"]
        )

        # Flatten for easier processing: (T, n_agents) -> (T * n_agents,)
        T, n_agents = rewards.shape
        obs_flat = observations.view(T * n_agents, -1)
        actions_flat = actions.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0

        # Multiple epochs
        for _ in range(self.config["num_epochs"]):
            # Get current policy output
            loc, scale = self.policy(obs_flat)
            dist = torch.distributions.Normal(loc, scale)
            new_log_probs_flat = dist.log_prob(actions_flat).sum(dim=-1)
            entropy = dist.entropy().mean()

            # Policy loss (PPO clip)
            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - self.config["clip_epsilon"], 1 + self.config["clip_epsilon"]) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            new_values = self.critic(obs_flat).squeeze(-1)
            value_loss = nn.functional.mse_loss(new_values, returns_flat)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.config["entropy_coef"] * entropy

            # Update
            self.policy_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.policy_optim.step()
            self.critic_optim.step()

            total_loss += loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()

        self.metrics["losses"].append(total_loss / self.config["num_epochs"])
        self.metrics["policy_losses"].append(policy_loss_total / self.config["num_epochs"])
        self.metrics["value_losses"].append(value_loss_total / self.config["num_epochs"])

        return total_loss / self.config["num_epochs"]

    def train(self, total_frames):
        """Main training loop."""
        frames_collected = 0
        update_count = 0

        while frames_collected < total_frames:
            # Collect rollout
            rollout = self.collect_rollout(self.config["frames_per_batch"])
            frames_collected += self.config["frames_per_batch"] * self.env.n_agents

            # Update
            loss = self.update(rollout)
            update_count += 1

            # Print progress
            if update_count % 5 == 0:
                recent_rewards = self.metrics["episode_rewards"][-10:] if self.metrics["episode_rewards"] else [0]
                recent_gini = self.metrics["gini_coefficients"][-10:] if self.metrics["gini_coefficients"] else [0]
                print(f"  Frames: {frames_collected}/{total_frames} | "
                      f"Loss: {loss:.4f} | "
                      f"Reward: {np.mean(recent_rewards):.2f} | "
                      f"Gini: {np.mean(recent_gini):.3f}")

        return self.metrics


# ============================================================
# Communication Training
# ============================================================

def train_communication_comparison(config):
    """Train and compare baseline vs communication."""
    print("\n" + "="*60)
    print("COMMUNICATION COMPARISON TRAINING")
    print("="*60)

    results = {"baseline": [], "discrete_comm": []}

    for seed in config["seeds"]:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Baseline (no communication)
        print("\nTraining BASELINE (no communication)...")
        env_baseline = MingleEnv(
            n_agents=config["n_agents"],
            n_rooms=config["n_rooms"],
            room_capacity=config["room_capacity"],
            max_steps=100,
        )

        trainer_baseline = SimplePPOTrainer(env_baseline, config, DEVICE)
        metrics_baseline = trainer_baseline.train(config["total_frames"])

        results["baseline"].append({
            "seed": seed,
            "rewards": metrics_baseline["rewards"],
            "episode_rewards": metrics_baseline["episode_rewards"],
            "losses": metrics_baseline["losses"],
            "gini_coefficient": np.mean(metrics_baseline["gini_coefficients"][-20:]) if metrics_baseline["gini_coefficients"] else 0,
            "reward_variance": np.var([r.mean() for r in metrics_baseline["agent_rewards"][-20:]]) if metrics_baseline["agent_rewards"] else 0,
            "final_reward": np.mean(metrics_baseline["episode_rewards"][-20:]) if metrics_baseline["episode_rewards"] else 0,
        })

        # Discrete Communication
        print("\nTraining DISCRETE COMMUNICATION...")
        try:
            from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
            env_comm = MingleEnvWithComm(
                n_agents=config["n_agents"],
                n_rooms=config["n_rooms"],
                room_capacity=config["room_capacity"],
                max_steps=100,
                vocab_size=2,
            )

            trainer_comm = SimplePPOTrainer(env_comm, config, DEVICE)
            metrics_comm = trainer_comm.train(config["total_frames"])

            results["discrete_comm"].append({
                "seed": seed,
                "rewards": metrics_comm["rewards"],
                "episode_rewards": metrics_comm["episode_rewards"],
                "losses": metrics_comm["losses"],
                "gini_coefficient": np.mean(metrics_comm["gini_coefficients"][-20:]) if metrics_comm["gini_coefficients"] else 0,
                "reward_variance": np.var([r.mean() for r in metrics_comm["agent_rewards"][-20:]]) if metrics_comm["agent_rewards"] else 0,
                "final_reward": np.mean(metrics_comm["episode_rewards"][-20:]) if metrics_comm["episode_rewards"] else 0,
            })
        except Exception as e:
            print(f"  Error training communication: {e}")
            results["discrete_comm"].append(None)

    # Save results
    save_path = RESULTS_DIR / "communication" / "comparison_results.json"
    save_results(results, save_path)

    # Generate plots
    plot_communication_results(results)

    return results


def plot_communication_results(results):
    """Generate communication comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Final Reward Comparison
    ax = axes[0, 0]
    categories = ["Baseline", "Discrete Comm"]

    baseline_rewards = [r["final_reward"] for r in results["baseline"] if r]
    comm_rewards = [r["final_reward"] for r in results["discrete_comm"] if r]

    means = [np.mean(baseline_rewards) if baseline_rewards else 0,
             np.mean(comm_rewards) if comm_rewards else 0]
    stds = [np.std(baseline_rewards) if baseline_rewards else 0,
            np.std(comm_rewards) if comm_rewards else 0]

    x = np.arange(len(categories))
    ax.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Final Reward Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Gini Coefficient
    ax = axes[0, 1]
    baseline_gini = [r["gini_coefficient"] for r in results["baseline"] if r]
    comm_gini = [r["gini_coefficient"] for r in results["discrete_comm"] if r]

    gini_means = [np.mean(baseline_gini) if baseline_gini else 0,
                  np.mean(comm_gini) if comm_gini else 0]
    gini_stds = [np.std(baseline_gini) if baseline_gini else 0,
                 np.std(comm_gini) if comm_gini else 0]

    ax.bar(x, gini_means, yerr=gini_stds, capsize=5, color=['steelblue', 'coral'])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Learning Curves (Rewards)
    ax = axes[1, 0]
    for label, data, color in [("Baseline", results["baseline"], "steelblue"),
                                ("Discrete Comm", results["discrete_comm"], "coral")]:
        if data and data[0]:
            rewards = data[0].get("episode_rewards", [])
            if rewards:
                window = max(1, len(rewards) // 20)
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Loss Curves
    ax = axes[1, 1]
    for label, data, color in [("Baseline", results["baseline"], "steelblue"),
                                ("Discrete Comm", results["discrete_comm"], "coral")]:
        if data and data[0]:
            losses = data[0].get("losses", [])
            if losses:
                ax.plot(losses, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "communication_comparison.png", dpi=150)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR / 'communication_comparison.png'}")


# ============================================================
# Fairness Training
# ============================================================

def train_fairness_comparison(config):
    """Train and compare baseline vs fairness modes."""
    print("\n" + "="*60)
    print("FAIRNESS COMPARISON TRAINING")
    print("="*60)

    fairness_modes = [
        ("none", "Baseline", None),
        ("gini", "Gini", {"mode": "gini", "alpha": 0.5}),
        ("participation_variance", "Participation", {"mode": "participation_variance", "alpha": 0.5}),
    ]

    results = {mode: [] for mode, _, _ in fairness_modes}

    for seed in config["seeds"]:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        for mode, name, fairness_cfg in fairness_modes:
            print(f"\nTraining {name}...")

            env = MingleEnv(
                n_agents=config["n_agents"],
                n_rooms=config["n_rooms"],
                room_capacity=config["room_capacity"],
                max_steps=100,
            )

            # Create fairness transform if needed
            transform = None
            if fairness_cfg:
                transform = make_fairness_reward_transform(fairness_cfg)
                # Initialize transform
                init_td = TensorDict({"observation": torch.randn(config["n_agents"], 14)}, batch_size=[])
                transform(init_td)

            trainer = SimplePPOTrainer(env, config, DEVICE, fairness_transform=transform)
            metrics = trainer.train(config["total_frames"])

            results[mode].append({
                "seed": seed,
                "rewards": metrics["rewards"],
                "episode_rewards": metrics["episode_rewards"],
                "losses": metrics["losses"],
                "gini_coefficient": np.mean(metrics["gini_coefficients"][-20:]) if metrics["gini_coefficients"] else 0,
                "reward_variance": np.var([r.mean() for r in metrics["agent_rewards"][-20:]]) if metrics["agent_rewards"] else 0,
                "final_reward": np.mean(metrics["episode_rewards"][-20:]) if metrics["episode_rewards"] else 0,
            })

    # Save results
    save_path = RESULTS_DIR / "fairness" / "comparison_results.json"
    save_results(results, save_path)

    # Generate plots
    plot_fairness_results(results, fairness_modes)

    return results


def plot_fairness_results(results, fairness_modes):
    """Generate fairness comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = [name for _, name, _ in fairness_modes]
    colors = ['steelblue', 'coral', 'seagreen']

    # Plot 1: Final Reward
    ax = axes[0, 0]
    means = []
    stds = []
    for mode, _, _ in fairness_modes:
        rewards = [r["final_reward"] for r in results[mode] if r]
        means.append(np.mean(rewards) if rewards else 0)
        stds.append(np.std(rewards) if rewards else 0)

    x = np.arange(len(categories))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(categories)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward by Fairness Mode")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Gini Coefficient
    ax = axes[0, 1]
    gini_means = []
    gini_stds = []
    for mode, _, _ in fairness_modes:
        gini = [r["gini_coefficient"] for r in results[mode] if r]
        gini_means.append(np.mean(gini) if gini else 0)
        gini_stds.append(np.std(gini) if gini else 0)

    ax.bar(x, gini_means, yerr=gini_stds, capsize=5, color=colors[:len(categories)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness by Mode")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Reward Variance
    ax = axes[1, 0]
    var_means = []
    var_stds = []
    for mode, _, _ in fairness_modes:
        var = [r["reward_variance"] for r in results[mode] if r]
        var_means.append(np.mean(var) if var else 0)
        var_stds.append(np.std(var) if var else 0)

    ax.bar(x, var_means, yerr=var_stds, capsize=5, color=colors[:len(categories)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Reward Variance")
    ax.set_title("Reward Distribution Variance")
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Learning Curves
    ax = axes[1, 1]
    for i, (mode, name, _) in enumerate(fairness_modes):
        if results[mode] and results[mode][0]:
            rewards = results[mode][0].get("episode_rewards", [])
            if rewards:
                window = max(1, len(rewards) // 20)
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fairness_comparison.png", dpi=150)
    plt.close()
    print(f"Plot saved to {PLOTS_DIR / 'fairness_comparison.png'}")


# ============================================================
# Utilities
# ============================================================

def save_results(results, path):
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train comparison experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--communication", action="store_true", help="Communication comparison")
    parser.add_argument("--fairness", action="store_true", help="Fairness comparison")
    parser.add_argument("--frames", type=int, default=10000, help="Total training frames")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds")

    args = parser.parse_args()

    if not (args.communication or args.fairness):
        args.all = True

    config = DEFAULT_CONFIG.copy()
    config["total_frames"] = args.frames
    config["seeds"] = args.seeds

    ensure_dirs()

    print("="*60)
    print("COOPERATIVE MINGLE - COMPARISON TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Total frames: {config['total_frames']}")
    print(f"Seeds: {config['seeds']}")

    if args.all or args.communication:
        train_communication_comparison(config)

    if args.all or args.fairness:
        train_fairness_comparison(config)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
