"""
Algorithm Comparison: PPO vs IPPO vs MAPPO

This script implements THREE DISTINCT ALGORITHMS with proper architectural differences:

1. PPO (Baseline): Single shared policy, single shared critic
   - All agents use the SAME network
   - Simple but no multi-agent awareness

2. IPPO (Independent PPO): Independent policies AND critics per agent
   - Each agent has its OWN policy and critic
   - No information sharing between agents

3. MAPPO (Multi-Agent PPO): Shared policy, CENTRALIZED critic
   - Shared policy for consistency
   - Critic sees ALL agents' observations for better value estimation
   - The key innovation for multi-agent learning

Usage:
    python train_algorithms.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    StayInRoomReward,
    GetToRoomReward,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("contribution_tests_and_comparisions/algorithms")


# ============================================================
# Observation Normalizer (Critical for stable training!)
# ============================================================

class ObservationNormalizer:
    """Running observation normalizer for stable training."""
    def __init__(self, obs_dim, device):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 0
        self.device = device

    def update(self, obs):
        """Update running statistics."""
        if obs.numel() == 0:
            return
        obs_flat = obs.reshape(-1, obs.shape[-1])
        batch_count = obs_flat.shape[0]
        if batch_count < 2:
            return

        batch_mean = obs_flat.mean(dim=0)
        batch_var = obs_flat.var(dim=0, unbiased=False) + 1e-8

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total_count
            self.var = (self.var * self.count + batch_var * batch_count +
                       delta**2 * self.count * batch_count / total_count) / total_count
            self.count = total_count

    def normalize(self, obs):
        """Normalize observation."""
        if self.count == 0:
            return obs
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


# ============================================================
# Shared Components
# ============================================================

class GoalConditionedPolicy(nn.Module):
    """Goal-conditioned policy that naturally moves toward rooms."""

    def __init__(self, obs_dim, action_dim=2, hidden_dim=128, n_agents=6):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # Agent embeddings for diverse behavior
        self.agent_embed_dim = 8
        self.agent_embedding = nn.Embedding(n_agents, self.agent_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + self.agent_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.3)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0)
        nn.init.normal_(self.agent_embedding.weight, mean=0, std=0.5)

    def forward(self, obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim) or (n_agents, obs_dim) or (batch*n_agents, obs_dim)
        """
        original_shape = obs.shape

        # Handle different input shapes
        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                # (n_agents, obs_dim) -> (1, n_agents, obs_dim)
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                # Flat batch: (batch*n_agents, obs_dim) -> (batch, n_agents, obs_dim)
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
            else:
                # Unknown flat shape - create agent IDs cyclically
                n_samples = obs.shape[0]
                agent_ids = torch.arange(n_samples, device=obs.device) % self.n_agents
                agent_embeds = self.agent_embedding(agent_ids)
                obs_with_id = torch.cat([obs, agent_embeds], dim=-1)
                hidden = self.net(obs_with_id)
                adjustment = torch.tanh(self.action_head(hidden))
                room_dir = obs[:, 5:7]
                loc = room_dir * 0.25 + adjustment * 0.05
                scale = self.log_std.exp().expand_as(loc)
                return loc, scale

        batch_size, n_agents, obs_dim = obs.shape

        # Add agent embeddings
        agent_ids = torch.arange(n_agents, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        agent_embeds = self.agent_embedding(agent_ids)
        obs_with_id = torch.cat([obs, agent_embeds], dim=-1)

        # Extract room direction (goal-conditioning)
        room_dir = obs[:, :, 5:7]

        # Through network
        obs_flat = obs_with_id.view(-1, obs_dim + self.agent_embed_dim)
        hidden = self.net(obs_flat)

        # Action adjustment bounded by tanh
        adjustment = torch.tanh(self.action_head(hidden))
        adjustment = adjustment.view(batch_size, n_agents, -1)

        # Goal-conditioned: room direction * 0.25 + adjustment * 0.05
        loc = room_dir * 0.25 + adjustment * 0.05
        scale = self.log_std.exp().expand_as(loc)

        # Return in appropriate shape
        if len(original_shape) == 2 and original_shape[0] % self.n_agents == 0 and original_shape[0] != self.n_agents:
            # Was flat batch, return flat
            return loc.view(-1, self.action_dim), scale.view(-1, self.action_dim)

        return loc.squeeze(0), scale.squeeze(0)


# ============================================================
# PPO: Single Shared Critic
# ============================================================

class PPOCritic(nn.Module):
    """Standard PPO critic - processes each agent independently."""

    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim) or (n_agents, obs_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        batch_size, n_agents, obs_dim = obs.shape
        obs_flat = obs.view(-1, obs_dim)
        values = self.net(obs_flat)
        return values.view(batch_size, n_agents, 1).squeeze(0)


# ============================================================
# IPPO: Independent Critics Per Agent
# ============================================================

class IPPOCritic(nn.Module):
    """IPPO - Each agent has its OWN critic network."""

    def __init__(self, obs_dim, n_agents, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents

        # Create independent critic for each agent
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_agents)
        ])

    def forward(self, obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim) or (n_agents, obs_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        batch_size, n_agents, obs_dim = obs.shape

        # Each agent uses its own critic
        values = []
        for i in range(n_agents):
            agent_obs = obs[:, i, :]  # (batch, obs_dim)
            agent_value = self.critics[i](agent_obs)  # (batch, 1)
            values.append(agent_value)

        values = torch.stack(values, dim=1)  # (batch, n_agents, 1)
        return values.squeeze(0)


# ============================================================
# MAPPO: Centralized Critic
# ============================================================

class MAPPOCritic(nn.Module):
    """MAPPO - Centralized critic that sees ALL agents' observations."""

    def __init__(self, obs_dim, n_agents, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents

        # Takes CONCATENATED observations from ALL agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),  # Value for each agent
        )

    def forward(self, obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim) or (n_agents, obs_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]

        # Concatenate ALL agents' observations (the key difference!)
        obs_concat = obs.view(batch_size, -1)

        values = self.net(obs_concat)  # (batch, n_agents)
        return values.unsqueeze(-1).squeeze(0)  # (n_agents, 1) or (batch, n_agents, 1)


# ============================================================
# Training Function
# ============================================================

def get_reward_modules():
    """Create reward modules."""
    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=3.0, phase_mode="claiming"),
    ]
    for m in modules:
        m._activate()
    return modules


def compute_gini(values):
    """Compute Gini coefficient."""
    values = np.array(values).flatten()
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.abs(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def train_algorithm(algorithm, config):
    """Train a specific algorithm with proper exploration and normalization."""
    print(f"\n{'='*60}")
    print(f"Training: {algorithm.upper()}")
    print(f"Environment: {config['n_agents']} agents, {config['n_rooms']} rooms")
    print(f"Total frames: {config['total_frames']}")
    print(f"{'='*60}")

    # Create environment
    env = MingleEnv(
        n_agents=config["n_agents"],
        n_rooms=config["n_rooms"],
        room_capacity=config["room_capacity"],
        max_steps=300,
        phase_mode="claiming",
        reward_modules=get_reward_modules(),
    )

    obs_dim = env.observation_spec["observation"].shape[-1]
    n_agents = config["n_agents"]

    # Create policy (shared for all algorithms)
    policy = GoalConditionedPolicy(obs_dim, n_agents=n_agents).to(DEVICE)

    # Create critic based on algorithm
    if algorithm == "ppo":
        critic = PPOCritic(obs_dim).to(DEVICE)
        critic_type = "Shared Critic"
    elif algorithm == "ippo":
        critic = IPPOCritic(obs_dim, n_agents).to(DEVICE)
        critic_type = "Independent Critics (one per agent)"
    elif algorithm == "mappo":
        critic = MAPPOCritic(obs_dim, n_agents).to(DEVICE)
        critic_type = "Centralized Critic (sees all agents)"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Critic type: {critic_type}")

    # OBSERVATION NORMALIZER - Critical for stable training!
    obs_normalizer = ObservationNormalizer(obs_dim, DEVICE)

    # Optimizers
    policy_optim = torch.optim.AdamW(policy.parameters(), lr=config["lr"], weight_decay=0.01)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config["lr"], weight_decay=0.01)

    # Metrics
    metrics = {
        "algorithm": algorithm,
        "rewards": [],
        "episode_rewards": [],
        "losses": [],
        "policy_losses": [],
        "value_losses": [],
        "gini_coefficients": [],
        "agent_rewards": [],
    }

    # Training loop
    frames = 0
    start_time = time.time()

    while frames < config["total_frames"]:
        # Collect rollout
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        episode_reward = 0
        agent_episode_rewards = torch.zeros(n_agents)

        # Epsilon for exploration (decays over training)
        epsilon = max(0.1, 0.5 * (1 - frames / config["total_frames"]))

        for _ in range(config["frames_per_batch"]):
            # Update normalizer and normalize
            obs_normalizer.update(obs_raw.unsqueeze(0))
            obs = obs_normalizer.normalize(obs_raw)

            with torch.no_grad():
                loc, scale = policy(obs.unsqueeze(0))
                # Clamp std for stability
                scale = scale.clamp(0.2, 1.0)
                dist = torch.distributions.Normal(loc, scale)
                action = dist.sample()

                # EPSILON-GREEDY EXPLORATION - Critical!
                if torch.rand(1).item() < epsilon:
                    action = torch.rand_like(action) * 0.6 - 0.3  # Random in [-0.3, 0.3]

                action = action.clamp(-0.3, 0.3)
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = critic(obs.unsqueeze(0))

            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            reward = next_td["reward"].to(DEVICE)
            done = next_td["done"].any() or next_td["terminated"].any()

            observations.append(obs)
            actions.append(action)
            rewards.append(reward.squeeze())
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.squeeze())

            episode_reward += reward.sum().item()
            agent_episode_rewards += reward.squeeze().cpu()
            metrics["rewards"].append(reward.mean().item())

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["agent_rewards"].append(agent_episode_rewards.numpy().copy())
                metrics["gini_coefficients"].append(compute_gini(agent_episode_rewards.numpy()))

                episode_reward = 0
                agent_episode_rewards = torch.zeros(n_agents)
                td = env.reset()
                obs_raw = td["observation"].to(DEVICE)
            else:
                obs_raw = next_td["observation"].to(DEVICE)

        frames += config["frames_per_batch"] * n_agents

        # Compute GAE
        obs_tensor = torch.stack(observations)
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.stack(rewards)
        dones_tensor = torch.tensor(dones, device=DEVICE)
        old_log_probs = torch.stack(log_probs)
        values_tensor = torch.stack(values)

        with torch.no_grad():
            next_value = critic(obs.unsqueeze(0)).squeeze()

        # GAE computation
        advantages = torch.zeros_like(rewards_tensor)
        gae = torch.zeros(n_agents, device=DEVICE)

        for t in reversed(range(len(rewards_tensor))):
            if t == len(rewards_tensor) - 1:
                next_val = next_value
            else:
                next_val = values_tensor[t + 1]

            done_mask = 1 - dones_tensor[t].float()
            delta = rewards_tensor[t] + config["gamma"] * next_val * done_mask - values_tensor[t]
            gae = delta + config["gamma"] * config["lmbda"] * done_mask * gae
            advantages[t] = gae

        returns = advantages + values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten
        T = len(rewards_tensor)
        obs_flat = obs_tensor.view(T * n_agents, -1)
        actions_flat = actions_tensor.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(config["num_epochs"]):
            loc, scale = policy(obs_flat)
            scale = scale.clamp(0.1, 1.0)
            dist = torch.distributions.Normal(loc, scale)
            new_log_probs = dist.log_prob(actions_flat).sum(dim=-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - config["clip_epsilon"], 1 + config["clip_epsilon"]) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = critic(obs_flat.view(T, n_agents, -1)).view(-1)
            value_loss = nn.functional.mse_loss(new_values, returns_flat)

            # Total loss with high entropy for exploration
            loss = policy_loss + 0.5 * value_loss - config["entropy_coef"] * entropy

            policy_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            policy_optim.step()
            critic_optim.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        metrics["losses"].append(total_loss / config["num_epochs"])
        metrics["policy_losses"].append(total_policy_loss / config["num_epochs"])
        metrics["value_losses"].append(total_value_loss / config["num_epochs"])

        # Progress
        if len(metrics["episode_rewards"]) > 0 and frames % 20000 < config["frames_per_batch"] * n_agents:
            recent_rewards = metrics["episode_rewards"][-10:] if metrics["episode_rewards"] else [0]
            recent_gini = metrics["gini_coefficients"][-10:] if metrics["gini_coefficients"] else [0]
            fps = frames / (time.time() - start_time)
            print(f"Frames: {frames:8d}/{config['total_frames']} | "
                  f"P_Loss: {total_policy_loss/config['num_epochs']:.3f} | "
                  f"V_Loss: {total_value_loss/config['num_epochs']:.3f} | "
                  f"Reward: {np.mean(recent_rewards):8.2f} | "
                  f"Gini: {np.mean(recent_gini):.3f} | "
                  f"Eps: {epsilon:.2f}")

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time:.1f}s")

    # Save model
    output_dir = BASE_DIR / algorithm
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "algorithm": algorithm,
        "obs_normalizer_mean": obs_normalizer.mean,
        "obs_normalizer_var": obs_normalizer.var,
    }, output_dir / "model.pt")
    print(f"Model saved: {output_dir / 'model.pt'}")

    # Generate GIF with observation normalizer
    generate_gif(env, policy, output_dir / "trained_policy.gif", f"{algorithm.upper()}", n_agents, obs_normalizer)

    return metrics, policy, env, obs_normalizer


def generate_gif(env, policy, output_path, title, n_agents, obs_normalizer=None):
    """Generate GIF of trained policy with exploration."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        for step in range(300):
            # Normalize observations if normalizer provided
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs_raw)
            else:
                obs = obs_raw

            with torch.no_grad():
                loc, scale = policy(obs.unsqueeze(0))
                # Add some exploration noise for interesting behavior
                noise = torch.randn_like(loc) * scale * 0.3
                actions = (loc + noise).clamp(-0.3, 0.3)

            step_td = TensorDict({"action": actions.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            # Render
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')
            ax.set_title(f"{title} | Step {step}", fontsize=14)

            ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))

            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                room_occupancies = env.room_occupancy.cpu().numpy()
                for i, (room_pos, occ) in enumerate(zip(room_positions, room_occupancies)):
                    color = "green" if occ >= env.room_capacity else "yellow" if occ > 0 else "lightgreen"
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=True, color=color, alpha=0.3))
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=False, color="green"))
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occ)}/{env.room_capacity}",
                            ha='center', va='center', fontsize=10, fontweight='bold')

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
            obs_raw = next_td["observation"].to(DEVICE)

        imageio.mimsave(str(output_path), gif_frames, fps=10, loop=0)
        print(f"GIF saved: {output_path}")
    except Exception as e:
        print(f"GIF generation failed: {e}")


def create_comparison_plots(all_metrics):
    """Create comparison plots for all algorithms."""
    comparison_dir = BASE_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    algorithms = list(all_metrics.keys())
    colors = {"ppo": "steelblue", "ippo": "coral", "mappo": "seagreen"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison: PPO vs IPPO vs MAPPO", fontsize=16, fontweight='bold')

    # Plot 1: Final Reward
    ax = axes[0, 0]
    rewards = [np.mean(all_metrics[alg]["episode_rewards"][-20:]) if all_metrics[alg]["episode_rewards"] else 0 for alg in algorithms]
    bars = ax.bar(algorithms, rewards, color=[colors[a] for a in algorithms], edgecolor='black')
    ax.set_ylabel("Mean Episode Reward (last 20)")
    ax.set_title("Final Performance")
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{r:.0f}', ha='center', fontweight='bold')

    # Plot 2: Gini Coefficient (Fairness)
    ax = axes[0, 1]
    gini = [np.mean(all_metrics[alg]["gini_coefficients"][-20:]) if all_metrics[alg]["gini_coefficients"] else 0 for alg in algorithms]
    bars = ax.bar(algorithms, gini, color=[colors[a] for a in algorithms], edgecolor='black')
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)
    for bar, g in zip(bars, gini):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{g:.3f}', ha='center', fontweight='bold')

    # Plot 3: Learning Curves
    ax = axes[1, 0]
    for alg in algorithms:
        rewards = all_metrics[alg]["episode_rewards"]
        if rewards:
            window = max(1, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=alg.upper(), color=colors[alg], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Loss Curves
    ax = axes[1, 1]
    for alg in algorithms:
        losses = all_metrics[alg]["losses"]
        if losses:
            ax.plot(losses, label=alg.upper(), color=colors[alg], linewidth=1, alpha=0.8)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_dir / "algorithm_comparison.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved: {comparison_dir / 'algorithm_comparison.png'}")

    # Save summary
    summary = {}
    for alg in algorithms:
        m = all_metrics[alg]
        summary[alg] = {
            "final_reward": np.mean(m["episode_rewards"][-20:]) if m["episode_rewards"] else 0,
            "peak_reward": np.max(m["episode_rewards"]) if m["episode_rewards"] else 0,
            "gini_coefficient": np.mean(m["gini_coefficients"][-20:]) if m["gini_coefficients"] else 0,
            "total_episodes": len(m["episode_rewards"]),
        }

    with open(comparison_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {comparison_dir / 'summary.json'}")


def save_metrics_json(metrics, output_dir):
    """Save metrics to JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    metrics_clean = {k: convert(v) for k, v in metrics.items()}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")


def create_evaluation_plot(metrics, output_dir, algorithm):
    """Create evaluation plots for a single algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{algorithm.upper()} Training Evaluation", fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    rewards = metrics["episode_rewards"]
    if rewards:
        ax.plot(rewards, alpha=0.3, color='blue')
        if len(rewards) > 10:
            window = max(1, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, color='blue', linewidth=2, label='Smoothed')
        ax.axhline(np.mean(rewards[-20:]), color='red', linestyle='--', label=f'Final: {np.mean(rewards[-20:]):.0f}')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Gini Coefficient
    ax = axes[0, 1]
    gini = metrics["gini_coefficients"]
    if gini:
        ax.plot(gini, alpha=0.5, color='green')
        if len(gini) > 10:
            window = max(1, len(gini) // 20)
            smoothed = np.convolve(gini, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, color='green', linewidth=2)
        ax.axhline(np.mean(gini[-20:]), color='red', linestyle='--', label=f'Final: {np.mean(gini[-20:]):.3f}')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Fairness (Lower = Better)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Training Loss
    ax = axes[1, 0]
    losses = metrics["losses"]
    if losses:
        ax.plot(losses, color='red', alpha=0.7)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.3)

    # Plot 4: Per-Agent Reward Distribution
    ax = axes[1, 1]
    agent_rewards = metrics["agent_rewards"]
    if agent_rewards and len(agent_rewards) > 20:
        last_rewards = np.array(agent_rewards[-20:])
        agent_means = last_rewards.mean(axis=0)
        agent_stds = last_rewards.std(axis=0)
        x = np.arange(len(agent_means))
        ax.bar(x, agent_means, yerr=agent_stds, capsize=5, color='steelblue', edgecolor='black')
        ax.set_xlabel("Agent ID")
        ax.set_ylabel("Mean Reward (Last 20 Episodes)")
        ax.set_title("Per-Agent Reward Distribution")
        ax.set_xticks(x)
        ax.set_xticklabels([f"A{i}" for i in x])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "evaluation.png", dpi=150)
    plt.close()
    print(f"Evaluation plot saved: {output_dir / 'evaluation.png'}")


def train_single_algorithm(algorithm, config):
    """Train a single algorithm with full evaluation."""
    metrics, policy, env, obs_normalizer = train_algorithm(algorithm, config)

    output_dir = BASE_DIR / algorithm
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    save_metrics_json(metrics, output_dir)

    # Create evaluation plot
    create_evaluation_plot(metrics, output_dir, algorithm)

    # Print summary
    print(f"\n{'='*50}")
    print(f"{algorithm.upper()} TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Total Episodes: {len(metrics['episode_rewards'])}")
    print(f"Final Reward (last 20): {np.mean(metrics['episode_rewards'][-20:]):.2f}")
    print(f"Peak Reward: {np.max(metrics['episode_rewards']):.2f}")
    print(f"Final Gini (last 20): {np.mean(metrics['gini_coefficients'][-20:]):.3f}")
    print(f"{'='*50}\n")

    return metrics


def main():
    """Run algorithm experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Train MARL algorithms")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "ippo", "mappo", "all"], default="all",
                       help="Which algorithm to train")
    parser.add_argument("--frames", type=int, default=300000, help="Total training frames")
    args = parser.parse_args()

    print("="*70)
    print("ALGORITHM TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "n_agents": 6,
        "n_rooms": 3,
        "room_capacity": 2,
        "total_frames": args.frames,
        "frames_per_batch": 2048,
        "num_epochs": 4,
        "lr": 3e-4,
        "gamma": 0.99,
        "lmbda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.1,
    }

    print(f"Total frames: {config['total_frames']}")

    if args.algorithm == "all":
        algorithms = ["ppo", "ippo", "mappo"]
    else:
        algorithms = [args.algorithm]

    all_metrics = {}

    for algorithm in algorithms:
        metrics = train_single_algorithm(algorithm, config)
        all_metrics[algorithm] = metrics

    # Create comparison plots if we trained all
    if len(all_metrics) == 3:
        create_comparison_plots(all_metrics)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
