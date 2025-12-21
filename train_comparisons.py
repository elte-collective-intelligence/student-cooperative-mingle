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
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    StayInRoomReward,
    GetToRoomReward,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("contribution_tests_and_comparisions")

DEFAULT_CONFIG = {
    "n_agents": 6,
    "n_rooms": 3,
    "room_capacity": 2,
    "total_frames": 300000,  # 300k frames for proper training
    "frames_per_batch": 2048,
    "num_epochs": 4,
    "minibatch_size": 256,
    "lr": 3e-4,
    "gamma": 0.99,
    "lmbda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.1,  # Higher for exploration
    "seeds": [0],  # Single seed for faster comparison
}


def ensure_dirs():
    """Create output directories."""
    BASE_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "communication").mkdir(exist_ok=True)
    (BASE_DIR / "communication" / "baseline").mkdir(exist_ok=True)
    (BASE_DIR / "communication" / "discrete_comm").mkdir(exist_ok=True)
    (BASE_DIR / "communication" / "comparison").mkdir(exist_ok=True)
    (BASE_DIR / "fairness").mkdir(exist_ok=True)
    (BASE_DIR / "fairness" / "baseline").mkdir(exist_ok=True)
    (BASE_DIR / "fairness" / "gini").mkdir(exist_ok=True)
    (BASE_DIR / "fairness" / "participation").mkdir(exist_ok=True)
    (BASE_DIR / "fairness" / "comparison").mkdir(exist_ok=True)


def generate_gif(env, policy, output_path, title="Trained Policy", n_agents=6):
    """Generate GIF showing trained policy behavior."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gif_frames = []
        td = env.reset()
        obs = td["observation"].to(DEVICE)

        for step in range(300):
            with torch.no_grad():
                loc, scale = policy(obs.unsqueeze(0))
                loc = loc.squeeze(0)
                actions = loc.clamp(-0.3, 0.3)

            step_td = TensorDict({"action": actions.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            # Render frame
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')
            ax.set_title(f"{title} | Step {step}", fontsize=14)

            # Arena
            ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))

            # Rooms
            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                room_occupancies = env.room_occupancy.cpu().numpy() if hasattr(env, 'room_occupancy') else [0] * len(room_positions)
                for i, (room_pos, occ) in enumerate(zip(room_positions, room_occupancies)):
                    color = "green" if occ >= env.room_capacity else "yellow" if occ > 0 else "lightgreen"
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=True, color=color, alpha=0.3))
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=False, color="green"))
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occ)}/{env.room_capacity}",
                            ha='center', va='center', fontsize=10, fontweight='bold')

            # Agents
            positions = env.agent_positions.cpu().numpy()
            colors_list = plt.cm.tab10(np.linspace(0, 1, n_agents))
            for i, pos in enumerate(positions):
                ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(f"A{i}", (pos[0], pos[1] + 0.7), ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            gif_frames.append(frame[..., :3])
            plt.close(fig)

            if next_td["done"].any() or next_td["terminated"].any():
                break
            obs = next_td["observation"].to(DEVICE)

        imageio.mimsave(output_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved: {output_path}")
    except Exception as e:
        print(f"GIF generation failed: {e}")


def save_metrics(metrics, path):
    """Save training metrics to JSON."""
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
        json.dump(convert(metrics), f, indent=2)
    print(f"Metrics saved: {path}")


def get_reward_modules():
    """Create strong reward modules for clear learning signal."""
    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=3.0, phase_mode="claiming"),
    ]
    for m in modules:
        m._activate()
    return modules


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
    """Goal-conditioned policy network - moves toward rooms by default."""
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # Agent embeddings for diverse behavior
        self.agent_embed_dim = 8
        self.agent_embedding = nn.Embedding(n_agents, self.agent_embed_dim)

        # Network outputs adjustment to room direction
        self.net = SimpleMLP(obs_dim + self.agent_embed_dim, action_dim * 2, hidden_dim)
        self.min_std = 0.1

        # Initialize agent embeddings
        nn.init.normal_(self.agent_embedding.weight, mean=0, std=0.5)

    def forward(self, obs):
        # obs shape: (batch, n_agents, obs_dim) or (n_agents, obs_dim) or (batch*n_agents, obs_dim)
        original_shape = obs.shape
        was_flat = False

        # Handle flat input by reshaping to 3D
        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                # (n_agents, obs_dim) -> (1, n_agents, obs_dim)
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                # (batch*n_agents, obs_dim) -> (batch, n_agents, obs_dim)
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
                was_flat = True
            else:
                # Unknown shape - use simple output
                # Create dummy agent IDs cycling through agents
                n_samples = obs.shape[0]
                agent_ids = torch.arange(n_samples, device=obs.device) % self.n_agents
                agent_embeds = self.agent_embedding(agent_ids)
                obs_with_id = torch.cat([obs, agent_embeds], dim=-1)
                out = self.net(obs_with_id)
                room_dir = obs[:, 5:7]
                adjustment = torch.tanh(out[..., :self.action_dim])
                loc = room_dir * 0.25 + adjustment * 0.05
                scale = torch.nn.functional.softplus(out[..., self.action_dim:]) + self.min_std
                return loc, scale

        batch_size, n_agents, obs_dim = obs.shape

        # Add agent embeddings
        agent_ids = torch.arange(n_agents, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        agent_embeds = self.agent_embedding(agent_ids)
        obs_with_id = torch.cat([obs, agent_embeds], dim=-1)

        # Extract room direction (indices 5-6 in base obs)
        room_dir = obs[:, :, 5:7]

        # Flatten for network
        obs_flat = obs_with_id.view(-1, obs_dim + self.agent_embed_dim)
        out = self.net(obs_flat)
        out = out.view(batch_size, n_agents, -1)

        # Split into adjustment and scale
        adjustment_raw = out[..., :self.action_dim]
        scale_raw = out[..., self.action_dim:]

        # Bound adjustment with tanh
        adjustment = torch.tanh(adjustment_raw)

        # Goal-conditioned: room_dir * 0.25 + adjustment * 0.05
        loc = room_dir * 0.25 + adjustment * 0.05
        scale = torch.nn.functional.softplus(scale_raw) + self.min_std

        # Reshape back if input was flat
        if was_flat:
            loc = loc.view(-1, self.action_dim)
            scale = scale.view(-1, self.action_dim)
        elif original_shape[0] == self.n_agents and len(original_shape) == 2:
            loc = loc.squeeze(0)
            scale = scale.squeeze(0)

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

        # Optimizers with weight decay
        self.policy_optim = torch.optim.AdamW(self.policy.parameters(), lr=config["lr"], weight_decay=0.01)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=config["lr"], weight_decay=0.01)

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
            max_steps=300,
            phase_mode="claiming",
            reward_modules=get_reward_modules(),
        )

        trainer_baseline = SimplePPOTrainer(env_baseline, config, DEVICE)
        metrics_baseline = trainer_baseline.train(config["total_frames"])

        # Save model and generate GIF
        baseline_dir = BASE_DIR / "communication" / "baseline"
        torch.save({"policy": trainer_baseline.policy.state_dict()}, baseline_dir / "model.pt")
        save_metrics(metrics_baseline, baseline_dir / "metrics.json")
        generate_gif(env_baseline, trainer_baseline.policy, baseline_dir / "trained_policy.gif",
                    "Communication: Baseline", config["n_agents"])

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
                max_steps=300,
                phase_mode="claiming",
                vocab_size=2,
                reward_modules=get_reward_modules(),
            )

            trainer_comm = SimplePPOTrainer(env_comm, config, DEVICE)
            metrics_comm = trainer_comm.train(config["total_frames"])

            # Save model and generate GIF
            comm_dir = BASE_DIR / "communication" / "discrete_comm"
            torch.save({"policy": trainer_comm.policy.state_dict()}, comm_dir / "model.pt")
            save_metrics(metrics_comm, comm_dir / "metrics.json")
            generate_gif(env_comm, trainer_comm.policy, comm_dir / "trained_policy.gif",
                        "Communication: Discrete Comm", config["n_agents"])

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
    save_path = BASE_DIR / "communication" / "comparison" / "comparison_results.json"
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
    plot_path = BASE_DIR / "communication" / "comparison" / "communication_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


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
                max_steps=300,
                phase_mode="claiming",
                reward_modules=get_reward_modules(),
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

            # Save model and generate GIF
            mode_folder = "baseline" if mode == "none" else ("participation" if mode == "participation_variance" else mode)
            method_dir = BASE_DIR / "fairness" / mode_folder
            torch.save({"policy": trainer.policy.state_dict()}, method_dir / "model.pt")
            save_metrics(metrics, method_dir / "metrics.json")
            generate_gif(env, trainer.policy, method_dir / "trained_policy.gif",
                        f"Fairness: {name}", config["n_agents"])

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
    save_path = BASE_DIR / "fairness" / "comparison" / "comparison_results.json"
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
    plot_path = BASE_DIR / "fairness" / "comparison" / "fairness_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


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
