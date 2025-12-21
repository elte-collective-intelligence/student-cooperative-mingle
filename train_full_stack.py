"""
FULL STACK Training: MAPPO + Gini Fairness + Discrete Communication

This combines all three contributions:
1. MAPPO: Multi-Agent PPO with shared centralized critic
2. Gini Fairness: Reward redistribution for equal outcomes
3. Discrete Communication: Agents can signal "Follow me" or "Room full"
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

from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    StayInRoomReward,
    GetToRoomReward,
)


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

        # Flatten to (batch * agents, obs_dim)
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
            return obs  # Don't normalize until we have statistics
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


class FullStackPolicy(nn.Module):
    """
    Policy network for Full Stack: handles movement + message output.

    Output: mean, std for movement actions + message logits
    """
    def __init__(self, obs_dim, action_dim=2, vocab_size=2, hidden_dim=128, n_agents=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_agents = n_agents

        # Input includes agent ID embedding
        self.agent_embed_dim = 8
        self.agent_embedding = nn.Embedding(n_agents, self.agent_embed_dim)

        # Shared backbone with LayerNorm for stability
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + self.agent_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Movement head (continuous actions)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        # Start with higher std for more exploration
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * 0.5)

        # Message head (discrete action)
        self.message_logits = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights for better exploration
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to encourage exploration."""
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller init
                nn.init.constant_(module.bias, 0)

        # Very small init for action mean to prevent tanh saturation
        nn.init.xavier_uniform_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, 0)

        # Message logits - equal probability initially
        nn.init.xavier_uniform_(self.message_logits.weight, gain=0.01)
        nn.init.constant_(self.message_logits.bias, 0)

        # Agent embedding - diverse initialization
        nn.init.normal_(self.agent_embedding.weight, mean=0, std=0.5)

    def forward(self, obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim) or (n_agents, obs_dim)
        Returns:
            action_mean, action_std, message_logits
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # Add batch dim

        batch_size, n_agents, obs_dim = obs.shape

        # Create agent IDs for embedding
        agent_ids = torch.arange(n_agents, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        agent_embeds = self.agent_embedding(agent_ids)  # (batch, n_agents, embed_dim)

        # Concatenate obs with agent embeddings
        obs_with_id = torch.cat([obs, agent_embeds], dim=-1)

        # Flatten for processing
        obs_flat = obs_with_id.view(-1, obs_dim + self.agent_embed_dim)

        # Through backbone
        hidden = self.backbone(obs_flat)

        # Movement output - goal-conditioned
        # Extract room direction from observation (indices 5-6 are nearest_room_dir)
        room_dir = obs[:, :, 5:7] if obs.dim() == 3 else obs[:, 5:7]
        room_dir_flat = room_dir.reshape(-1, 2)

        # Policy head outputs bounded adjustment [-1, 1]
        action_adjustment_raw = self.action_mean(hidden)
        action_adjustment = torch.tanh(action_adjustment_raw)  # Bound to [-1, 1]

        # Action = base direction toward room (main component) + small learned adjustment
        # 0.25 toward room, 0.05 adjustment = agents mostly follow room direction
        action_mean = room_dir_flat * 0.25 + action_adjustment * 0.05
        action_std = self.action_log_std.exp().expand_as(action_mean)

        # Message output
        msg_logits = self.message_logits(hidden)

        # Reshape back
        action_mean = action_mean.view(batch_size, n_agents, -1)
        action_std = action_std.view(batch_size, n_agents, -1)
        msg_logits = msg_logits.view(batch_size, n_agents, -1)

        return action_mean, action_std, msg_logits


class MAPPOCritic(nn.Module):
    """
    MAPPO Centralized Critic: sees ALL agents' observations.

    This is the key difference from IPPO - shared information for value estimation.
    """
    def __init__(self, obs_dim, n_agents, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents

        # Takes concatenated observations from ALL agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),  # Value for each agent
        )

    def forward(self, obs):
        """
        Args:
            obs: (batch, n_agents, obs_dim)
        Returns:
            values: (batch, n_agents, 1)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]

        # Concatenate all agents' observations (centralized)
        obs_concat = obs.view(batch_size, -1)

        # Get values for all agents
        values = self.net(obs_concat)  # (batch, n_agents)

        return values.unsqueeze(-1)  # (batch, n_agents, 1)


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - dones[t].float()) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t].float()) * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)
    returns = advantages + torch.stack(values) if isinstance(values, list) else advantages + values

    return advantages, returns


def train_full_stack(
    n_agents=6,
    n_rooms=3,
    total_frames=150000,
    output_dir="contribution_tests_and_comparisions/full_stack",
    device=None,
):
    """
    Train the Full Stack model: MAPPO + Gini Fairness + Discrete Communication.
    """
    print("\n" + "="*70)
    print("FULL STACK: MAPPO + Gini Fairness + Discrete Communication")
    print("="*70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Setup reward modules - STRONGER rewards for clear signal
    reward_modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),  # Much stronger
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=3.0, phase_mode="claiming"),
    ]
    for m in reward_modules:
        m._activate()

    # Create environment with DISCRETE COMMUNICATION
    env = MingleEnvWithComm(
        n_agents=n_agents,
        n_rooms=n_rooms,
        arena_radius=10.0,
        center_radius=3.0,
        max_steps=300,
        phase_mode="claiming",
        room_radius=2.0,
        room_capacity=2,
        vocab_size=2,  # "Follow me" and "Room full"
        comm_range=None,  # Global broadcast
        reward_modules=reward_modules,
    )

    # Create Gini fairness transform
    fairness_transform = make_fairness_reward_transform({
        "mode": "gini",
        "alpha": 0.5,
    })

    # Get dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = 2
    vocab_size = env.vocab_size

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Vocab size: {vocab_size}")
    print(f"Agents: {n_agents}, Rooms: {n_rooms}")

    # Create networks
    policy = FullStackPolicy(obs_dim, action_dim, vocab_size, hidden_dim=128, n_agents=n_agents).to(device)
    critic = MAPPOCritic(obs_dim, n_agents, hidden_dim=128).to(device)

    # Observation normalizer for stable training
    obs_normalizer = ObservationNormalizer(obs_dim, device)

    # Optimizers with weight decay to prevent weight explosion
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=3e-4, weight_decay=0.01)

    # Training parameters
    batch_size = 2048
    mini_batch_size = 256
    ppo_epochs = 4
    clip_eps = 0.2
    gamma = 0.99
    gae_lambda = 0.95

    # Metrics tracking
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "critic_losses": [],
        "message_usage": [],
    }

    # Training loop
    frames = 0
    episode = 0
    start_time = time.time()

    while frames < total_frames:
        # Collect batch of experience
        batch_obs = []
        batch_actions = []
        batch_messages = []
        batch_rewards = []
        batch_values = []
        batch_log_probs = []
        batch_msg_log_probs = []
        batch_dones = []

        batch_frames = 0
        episode_reward = 0
        episode_length = 0
        msg_counts = torch.zeros(vocab_size)

        td = env.reset()
        obs_raw = td["observation"].to(device)

        # Initialize fairness transform
        fairness_transform(td)

        # Epsilon for exploration (decays over training)
        epsilon = max(0.1, 0.5 * (1 - frames / total_frames))

        while batch_frames < batch_size:
            # Update normalizer and normalize observations
            obs_normalizer.update(obs_raw.unsqueeze(0))
            obs = obs_normalizer.normalize(obs_raw)

            with torch.no_grad():
                action_mean, action_std, msg_logits = policy(obs.unsqueeze(0))
                action_mean = action_mean.squeeze(0)
                action_std = action_std.squeeze(0)
                msg_logits = msg_logits.squeeze(0)

                # Sample actions with epsilon-greedy exploration
                action_dist = torch.distributions.Normal(action_mean, action_std.clamp(0.2, 1.0))
                actions = action_dist.sample()

                # Epsilon-greedy: with probability epsilon, take random action
                if torch.rand(1).item() < epsilon:
                    actions = torch.rand_like(actions) * 0.6 - 0.3  # Random in [-0.3, 0.3]

                actions = actions.clamp(-0.3, 0.3)
                action_log_prob = action_dist.log_prob(actions).sum(dim=-1)

                # Sample messages
                msg_dist = torch.distributions.Categorical(logits=msg_logits)
                messages = msg_dist.sample()
                msg_log_prob = msg_dist.log_prob(messages)

                # Get value estimate (MAPPO centralized critic)
                values = critic(obs.unsqueeze(0)).squeeze(0)

            # Step environment
            step_td = TensorDict({
                "action": actions.cpu(),
                "message": messages.cpu(),
            }, batch_size=[])
            next_td = env._step(step_td)

            # Apply Gini fairness transform to rewards
            next_td = fairness_transform(next_td)

            reward = next_td["reward"].to(device)
            done = next_td["done"].any() or next_td["terminated"].any()

            # Store experience
            batch_obs.append(obs)
            batch_actions.append(actions)
            batch_messages.append(messages)
            batch_rewards.append(reward)
            batch_values.append(values)
            batch_log_probs.append(action_log_prob)
            batch_msg_log_probs.append(msg_log_prob)
            batch_dones.append(torch.tensor([done] * n_agents, device=device))

            # Track metrics
            episode_reward += reward.sum().item()
            episode_length += 1
            batch_frames += n_agents
            frames += n_agents

            for m in messages:
                msg_counts[m.item()] += 1

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["episode_lengths"].append(episode_length)
                metrics["message_usage"].append(msg_counts.tolist())

                episode += 1
                episode_reward = 0
                episode_length = 0
                msg_counts = torch.zeros(vocab_size)

                td = env.reset()
                obs_raw = td["observation"].to(device)
                fairness_transform(td)
            else:
                obs_raw = next_td["observation"].to(device)

        # Compute advantages and returns
        obs = obs_normalizer.normalize(obs_raw)
        with torch.no_grad():
            next_values = critic(obs.unsqueeze(0)).squeeze(0)

        advantages_list = []
        returns_list = []
        gae = torch.zeros(n_agents, 1, device=device)

        for t in reversed(range(len(batch_rewards))):
            if t == len(batch_rewards) - 1:
                next_val = next_values
            else:
                next_val = batch_values[t + 1]

            delta = batch_rewards[t] + gamma * next_val * (1 - batch_dones[t].float().unsqueeze(1)) - batch_values[t]
            gae = delta + gamma * gae_lambda * (1 - batch_dones[t].float().unsqueeze(1)) * gae
            advantages_list.insert(0, gae.clone())
            returns_list.insert(0, gae + batch_values[t])

        # Stack tensors
        obs_batch = torch.stack(batch_obs)
        actions_batch = torch.stack(batch_actions)
        messages_batch = torch.stack(batch_messages)
        old_log_probs = torch.stack(batch_log_probs)
        old_msg_log_probs = torch.stack(batch_msg_log_probs)
        advantages = torch.stack(advantages_list).squeeze(-1)
        returns = torch.stack(returns_list)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_policy_loss = 0
        total_critic_loss = 0
        num_updates = 0

        for _ in range(ppo_epochs):
            # Random permutation
            indices = torch.randperm(len(obs_batch))

            for start in range(0, len(obs_batch), mini_batch_size // n_agents):
                end = min(start + mini_batch_size // n_agents, len(obs_batch))
                idx = indices[start:end]

                # Get minibatch
                mb_obs = obs_batch[idx]
                mb_actions = actions_batch[idx]
                mb_messages = messages_batch[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_old_msg_log_probs = old_msg_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Forward pass
                action_mean, action_std, msg_logits = policy(mb_obs)
                values = critic(mb_obs)

                # Action distribution
                action_dist = torch.distributions.Normal(action_mean, action_std.clamp(0.1, 1.0))
                new_log_probs = action_dist.log_prob(mb_actions).sum(dim=-1)

                # Message distribution
                msg_dist = torch.distributions.Categorical(logits=msg_logits)
                new_msg_log_probs = msg_dist.log_prob(mb_messages)

                # PPO ratio
                ratio = (new_log_probs - mb_old_log_probs).exp()
                msg_ratio = (new_msg_log_probs - mb_old_msg_log_probs).exp()

                # Clipped surrogate loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Message loss
                msg_surr1 = msg_ratio * mb_advantages
                msg_surr2 = torch.clamp(msg_ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
                msg_loss = -torch.min(msg_surr1, msg_surr2).mean()

                # Entropy bonus - HIGH for exploration!
                entropy = action_dist.entropy().mean() + msg_dist.entropy().mean()

                # Total policy loss - HIGH entropy coefficient for exploration
                entropy_coef = 0.1  # Much higher for exploration
                total_policy_loss_step = policy_loss + msg_loss - entropy_coef * entropy

                # Critic loss
                critic_loss = 0.5 * (values - mb_returns).pow(2).mean()

                # Update
                policy_optimizer.zero_grad()
                total_policy_loss_step.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

                total_policy_loss += total_policy_loss_step.item()
                total_critic_loss += critic_loss.item()
                num_updates += 1

        if num_updates > 0:
            metrics["policy_losses"].append(total_policy_loss / num_updates)
            metrics["critic_losses"].append(total_critic_loss / num_updates)

        # Print progress
        if len(metrics["episode_rewards"]) > 0 and len(metrics["episode_rewards"]) % 10 == 0:
            recent_rewards = metrics["episode_rewards"][-10:]
            fps = frames / (time.time() - start_time)
            print(f"Frames: {frames:6d}/{total_frames} | "
                  f"Episodes: {episode:4d} | "
                  f"Reward: {np.mean(recent_rewards):8.2f} | "
                  f"FPS: {fps:.0f}")

    # Save model
    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "config": {
            "n_agents": n_agents,
            "n_rooms": n_rooms,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "vocab_size": vocab_size,
        }
    }, os.path.join(output_dir, "model.pt"))

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Model saved: {os.path.join(output_dir, 'model.pt')}")

    # Generate GIF with normalizer
    generate_full_stack_gif(env, policy, device, output_dir, n_agents, obs_normalizer)

    # Plot training curves
    plot_training_curves(metrics, output_dir)

    return metrics


def generate_full_stack_gif(env, policy, device, output_dir, n_agents, obs_normalizer=None):
    """Generate GIF showing trained full stack policy."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(device)

        msg_meanings = {0: "Follow", 1: "Full"}

        for step in range(300):
            # Normalize observations if normalizer provided
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs_raw)
            else:
                obs = obs_raw

            with torch.no_grad():
                action_mean, action_std, msg_logits = policy(obs.unsqueeze(0))
                action_mean = action_mean.squeeze(0)
                action_std = action_std.squeeze(0)
                msg_logits = msg_logits.squeeze(0)

                # Add small exploration noise for diverse behavior
                action_std = action_std.squeeze(0)
                noise = torch.randn_like(action_mean) * action_std * 0.5
                actions = (action_mean + noise).clamp(-0.3, 0.3)

                msg_dist = torch.distributions.Categorical(logits=msg_logits)
                messages = msg_dist.sample()

            step_td = TensorDict({
                "action": actions.cpu(),
                "message": messages.cpu(),
            }, batch_size=[])
            next_td = env._step(step_td)

            # Render frame
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
            ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
            ax.set_aspect('equal')
            ax.set_title(f"FULL STACK: MAPPO + Gini + Comm | Step {step}", fontsize=14)

            # Arena
            ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))

            # Rooms with capacity
            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                room_occupancies = env.room_occupancy.cpu().numpy()
                for i, (room_pos, occ) in enumerate(zip(room_positions, room_occupancies)):
                    fill_ratio = min(occ / env.room_capacity, 1.0)
                    if fill_ratio >= 1.0:
                        color, alpha = "green", 0.4
                    elif fill_ratio > 0:
                        color, alpha = "yellow", 0.3
                    else:
                        color, alpha = "green", 0.2
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=True, color=color, alpha=alpha))
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=False, color="green"))
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occ)}/{env.room_capacity}",
                            ha='center', va='center', fontsize=10, fontweight='bold')

            # Agents
            positions = env.agent_positions.cpu().numpy()
            in_center = np.linalg.norm(positions, axis=1) <= env.center_radius
            in_room = np.zeros(len(positions), dtype=bool)

            if hasattr(env, "room_positions"):
                room_positions = env.room_positions.cpu().numpy()
                for i, pos in enumerate(positions):
                    if np.linalg.norm(room_positions - pos, axis=1).min() < env.room_radius:
                        in_room[i] = True

            outside = ~in_center & ~in_room

            # Color by status
            ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=100, label="In Room", zorder=5)
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=100, label="In Center", zorder=5)
            ax.scatter(positions[outside, 0], positions[outside, 1], c="orange", s=100, label="Outside", zorder=5)

            # Agent labels with messages
            for i, pos in enumerate(positions):
                msg = messages[i].item()
                msg_text = msg_meanings.get(msg, str(msg))
                ax.annotate(f"A{i}:{msg_text}", (pos[0], pos[1] + 0.6), ha='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.legend(loc='upper right')

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            gif_frames.append(frame[..., :3])
            plt.close(fig)

            done = next_td["done"].any() or next_td["terminated"].any()
            if done:
                break
            obs_raw = next_td["observation"].to(device)

        gif_path = os.path.join(output_dir, "full_stack_policy.gif")
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved: {gif_path}")

    except Exception as e:
        import traceback
        print(f"GIF generation failed: {e}")
        traceback.print_exc()


def plot_training_curves(metrics, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    ax = axes[0, 0]
    rewards = metrics["episode_rewards"]
    if len(rewards) > 10:
        window = min(50, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, linewidth=2, color='blue')
        ax.fill_between(range(len(smoothed)), smoothed - np.std(rewards[:len(smoothed)]),
                       smoothed + np.std(rewards[:len(smoothed)]), alpha=0.2)
    else:
        ax.plot(rewards, linewidth=2, color='blue')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards (Smoothed)")
    ax.grid(True, alpha=0.3)

    # Policy loss
    ax = axes[0, 1]
    if metrics["policy_losses"]:
        ax.plot(metrics["policy_losses"], linewidth=1, color='red', alpha=0.7)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[1, 0]
    if metrics["critic_losses"]:
        ax.plot(metrics["critic_losses"], linewidth=1, color='green', alpha=0.7)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Loss")
    ax.grid(True, alpha=0.3)

    # Message usage
    ax = axes[1, 1]
    if metrics["message_usage"]:
        msg_array = np.array(metrics["message_usage"])
        if len(msg_array) > 0:
            ax.plot(msg_array[:, 0] if msg_array.shape[1] > 0 else [], label="Follow me", linewidth=1)
            ax.plot(msg_array[:, 1] if msg_array.shape[1] > 1 else [], label="Room full", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Message Count")
    ax.set_title("Message Usage")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved: {os.path.join(output_dir, 'training_curves.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Full Stack Model")
    parser.add_argument("--frames", type=int, default=150000, help="Total training frames")
    parser.add_argument("--agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--rooms", type=int, default=3, help="Number of rooms")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_full_stack(
        n_agents=args.agents,
        n_rooms=args.rooms,
        total_frames=args.frames,
        device=device,
    )
