"""
Comprehensive Experiment Runner for Cooperative Mingle
Trains all experiments separately and saves results to contribution_tests_and_comparisions/

Usage:
    python run_all_experiments.py --all              # Run all experiments
    python run_all_experiments.py --algorithms       # Run only algorithm comparison
    python run_all_experiments.py --communication    # Run only communication comparison
    python run_all_experiments.py --fairness         # Run only fairness comparison
    python run_all_experiments.py --curriculum       # Run only curriculum learning
    python run_all_experiments.py --combined         # Run with all features combined
"""

import os
import sys
import json
import time
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform
from src.envs.modules.reward_module import (
    InsideCenterReward,
    CollisionAvoidanceReward,
    StayInRoomReward,
    LeaveToRoomReward,
    GetToRoomReward,
)


class PolicyNetwork(nn.Module):
    """Policy network for continuous actions."""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.min_std = 0.1

    def forward(self, obs):
        if obs.dim() == 3:
            batch, n_agents, obs_dim = obs.shape
            obs = obs.view(-1, obs_dim)
            h = self.net(obs)
            mean = self.mean(h)
            std = torch.exp(self.log_std(h)).clamp(min=self.min_std)
            return mean.view(batch, n_agents, -1), std.view(batch, n_agents, -1)
        h = self.net(obs)
        return self.mean(h), torch.exp(self.log_std(h)).clamp(min=self.min_std)


class CriticNetwork(nn.Module):
    """Value network."""
    def __init__(self, obs_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        if obs.dim() == 3:
            batch, n_agents, obs_dim = obs.shape
            obs = obs.view(-1, obs_dim)
            v = self.net(obs)
            return v.view(batch, n_agents, 1)
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """Centralized critic for MAPPO (takes all agent observations)."""
    def __init__(self, obs_dim, n_agents, hidden_dim=256, num_layers=2):
        super().__init__()
        input_dim = obs_dim * n_agents
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, n_agents))
        self.net = nn.Sequential(*layers)

    def forward(self, all_obs):
        # all_obs: [batch, n_agents, obs_dim] or [n_agents, obs_dim]
        if all_obs.dim() == 2:
            flat = all_obs.flatten().unsqueeze(0)
            return self.net(flat).squeeze(0).unsqueeze(-1)
        else:
            batch = all_obs.shape[0]
            flat = all_obs.view(batch, -1)
            return self.net(flat).unsqueeze(-1)


class ExperimentTrainer:
    """Unified trainer for all algorithms."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.algorithm = config.get("algorithm", "ppo")

        # Create reward modules - LINEAR GRADIENT toward rooms
        # Key fix: GetToRoomReward now provides smooth gradient, not flat reward
        reward_modules = [
            # Collision avoidance
            CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="claiming"),
            # LINEAR GRADIENT toward rooms - agents learn direction to rooms
            GetToRoomReward(max_reward=5.0, phase_mode="claiming"),
            # Reward for staying in room
            StayInRoomReward(max_reward=10.0, outside_penalty=1.0, overfill_penalty=1.5, phase_mode="claiming"),
        ]
        # Activate all reward modules
        for module in reward_modules:
            module._activate()

        # Create environment with reward modules
        # NO SPINNING PHASE - go straight to room-finding!
        self.env = MingleEnv(
            n_agents=config["n_agents"],
            n_rooms=config["n_rooms"],
            room_capacity=config.get("room_capacity", 2),
            arena_radius=config.get("arena_radius", 10.0),
            center_radius=config.get("center_radius", 3.0),
            room_radius=config.get("room_radius", 2.0),
            max_steps=config.get("max_steps", 300),  # Full 300 steps for room-finding
            phase_mode="claiming",  # NO SPINNING - straight to claiming!
            reward_modules=reward_modules,
        )

        # Fairness transform
        self.fairness_transform = None
        if config.get("fairness_mode") and config["fairness_mode"] != "none":
            self.fairness_transform = make_fairness_reward_transform({
                "mode": config["fairness_mode"],
                "alpha": config.get("fairness_alpha", 0.5),
            })

        # Get dimensions
        self.obs_dim = self.env.observation_spec["observation"].shape[-1]
        obs_dim = self.obs_dim  # Keep local reference for compatibility
        action_dim = 2
        n_agents = config["n_agents"]
        hidden_dim = config.get("hidden_dim", 128)

        # Create networks based on algorithm
        if self.algorithm == "ippo":
            # Independent networks for each agent
            self.policies = nn.ModuleList([
                PolicyNetwork(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)
            ]).to(device)
            self.critics = nn.ModuleList([
                CriticNetwork(obs_dim, hidden_dim) for _ in range(n_agents)
            ]).to(device)
            # Lower learning rate for IPPO to prevent divergence
            ippo_lr = config.get("lr", 3e-4) * 0.3
            self.policy_optim = torch.optim.Adam(self.policies.parameters(), lr=ippo_lr)
            self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=ippo_lr)
        elif self.algorithm == "mappo":
            # Shared policy, centralized critic
            self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
            self.critic = CentralizedCritic(obs_dim, n_agents, hidden_dim * 2).to(device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 3e-4))
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.get("lr", 3e-4))
        else:  # ppo
            self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
            self.critic = CriticNetwork(obs_dim, hidden_dim).to(device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 3e-4))
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.get("lr", 3e-4))

        # Training params
        self.gamma = config.get("gamma", 0.99)
        self.lmbda = config.get("lmbda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        # INCREASED entropy for more exploration (was 0.01, now 0.05)
        self.entropy_coef = config.get("entropy_coef", 0.05)
        self.value_coef = config.get("value_coef", 0.5)
        self.num_epochs = config.get("num_epochs", 4)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Metrics
        self.metrics = {
            "episode_rewards": [],
            "mean_rewards": [],
            "losses": [],
            "gini_coefficients": [],
        }

    def get_action(self, obs, agent_idx=None):
        """Get action from policy."""
        if self.algorithm == "ippo" and agent_idx is not None:
            mean, std = self.policies[agent_idx](obs)
        else:
            mean, std = self.policy(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(-self.env.max_speed, self.env.max_speed)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist

    def get_value(self, obs, all_obs=None):
        """Get value estimate."""
        if self.algorithm == "mappo" and all_obs is not None:
            return self.critic(all_obs)
        elif self.algorithm == "ippo":
            values = []
            n_agents = obs.shape[-2] if obs.dim() > 1 else self.config["n_agents"]
            for i in range(n_agents):
                agent_obs = obs[..., i, :] if obs.dim() > 1 else obs
                values.append(self.critics[i](agent_obs.unsqueeze(0)).squeeze(0))
            return torch.stack(values, dim=-2)
        else:
            return self.critic(obs)

    def collect_rollout(self, num_steps):
        """Collect experience."""
        observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        td = self.env.reset()
        obs = td["observation"].to(self.device)
        episode_reward = 0.0  # Track cumulative episode reward

        for _ in range(num_steps):
            with torch.no_grad():
                if self.algorithm == "ippo":
                    agent_actions, agent_log_probs = [], []
                    for i in range(self.config["n_agents"]):
                        mean, std = self.policies[i](obs[i:i+1])
                        mean = torch.clamp(mean, -10, 10)
                        std = torch.clamp(std, 0.1, 2.0)
                        dist = torch.distributions.Normal(mean.squeeze(0), std.squeeze(0))
                        action = dist.sample().clamp(-self.env.max_speed, self.env.max_speed)
                        lp = dist.log_prob(action).sum(dim=-1)
                        agent_actions.append(action)
                        agent_log_probs.append(lp)
                    action = torch.stack(agent_actions)
                    log_prob = torch.stack(agent_log_probs)
                else:
                    mean, std = self.policy(obs.unsqueeze(0))
                    mean, std = mean.squeeze(0), std.squeeze(0)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample().clamp(-self.env.max_speed, self.env.max_speed)
                    log_prob = dist.log_prob(action).sum(dim=-1)

                if self.algorithm == "mappo":
                    value = self.critic(obs.unsqueeze(0)).squeeze(0)
                elif self.algorithm == "ippo":
                    values_list = []
                    for i in range(self.config["n_agents"]):
                        v = self.critics[i](obs[i:i+1])
                        values_list.append(v.squeeze())
                    value = torch.stack(values_list)
                else:
                    value = self.critic(obs.unsqueeze(0)).squeeze(0)

            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            next_td = self.env._step(step_td)

            reward = next_td["reward"].to(self.device)

            # Apply fairness transform
            if self.fairness_transform is not None:
                transform_td = TensorDict({
                    "observation": next_td["observation"],
                    "reward": reward.cpu(),
                }, batch_size=[])
                transformed = self.fairness_transform(transform_td)
                reward = transformed["reward"].to(self.device)

            done = next_td["done"].any() or next_td["terminated"].any()

            observations.append(obs)
            actions.append(action)
            rewards.append(reward.squeeze())
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.squeeze())

            # Accumulate episode reward
            episode_reward += reward.sum().item()

            if done:
                self.metrics["episode_rewards"].append(episode_reward)
                episode_reward = 0.0  # Reset for next episode
                td = self.env.reset()
                obs = td["observation"].to(self.device)
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

    def compute_gae(self, rewards, values, dones):
        """Compute GAE."""
        T, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(n_agents, device=rewards.device)

        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else torch.zeros(n_agents, device=rewards.device)
            mask = 1 - dones[t].float()
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            advantages[t] = delta + self.gamma * self.lmbda * mask * last_adv
            last_adv = advantages[t]

        return advantages, advantages + values

    def update(self, rollout):
        """PPO update."""
        obs = rollout["observations"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]

        advantages, returns = self.compute_gae(rewards, values, dones)

        T, n_agents = rewards.shape
        obs_flat = obs.view(T * n_agents, -1)
        actions_flat = actions.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        total_loss = 0

        for _ in range(self.num_epochs):
            if self.algorithm == "ippo":
                policy_loss_total = 0
                value_loss_total = 0
                entropy_total = 0
                for i in range(n_agents):
                    agent_obs = obs[:, i]
                    agent_actions = actions[:, i]
                    agent_old_log_probs = old_log_probs[:, i]
                    agent_advantages = advantages[:, i]
                    agent_returns = returns[:, i]

                    agent_adv = (agent_advantages - agent_advantages.mean()) / (agent_advantages.std() + 1e-8)

                    mean, std = self.policies[i](agent_obs)
                    # Clamp values to avoid NaN
                    mean = torch.clamp(mean, -10, 10)
                    std = torch.clamp(std, 0.1, 2.0)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(agent_actions).sum(dim=-1)
                    entropy = dist.entropy().mean()
                    entropy_total += entropy

                    ratio = torch.exp(torch.clamp(new_log_probs - agent_old_log_probs, -20, 20))
                    surr1 = ratio * agent_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * agent_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss_total += policy_loss

                    new_values = self.critics[i](agent_obs).squeeze(-1)
                    value_loss = nn.functional.mse_loss(new_values, agent_returns)
                    value_loss_total += value_loss

                loss = policy_loss_total + self.value_coef * value_loss_total - self.entropy_coef * entropy_total
            else:
                mean, std = self.policy(obs_flat)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions_flat).sum(dim=-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_flat)
                surr1 = ratio * advantages_flat
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.algorithm == "mappo":
                    new_values = self.critic(obs).view(-1)
                else:
                    new_values = self.critic(obs_flat).squeeze(-1)
                value_loss = nn.functional.mse_loss(new_values, returns_flat)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.policy_optim.zero_grad()
            self.critic_optim.zero_grad()

            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping update")
                continue

            loss.backward()

            if self.algorithm == "ippo":
                # Stricter gradient clipping for IPPO
                nn.utils.clip_grad_norm_(self.policies.parameters(), 0.1)
                nn.utils.clip_grad_norm_(self.critics.parameters(), 0.1)
            else:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.policy_optim.step()
            self.critic_optim.step()
            total_loss += loss.item()

        self.metrics["losses"].append(total_loss / self.num_epochs)
        return total_loss / self.num_epochs

    def train(self, total_frames, frames_per_batch=2048, log_interval=10):
        """Main training loop."""
        frames = 0
        update_count = 0
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training: {self.algorithm.upper()}")
        print(f"Environment: {self.config['n_agents']} agents, {self.config['n_rooms']} rooms")
        print(f"Fairness: {self.config.get('fairness_mode', 'none')}")
        print(f"Total frames: {total_frames}")
        print(f"{'='*60}\n")

        while frames < total_frames:
            rollout = self.collect_rollout(frames_per_batch // self.config["n_agents"])
            frames += frames_per_batch
            loss = self.update(rollout)
            update_count += 1

            if len(self.metrics["episode_rewards"]) > 0:
                recent = self.metrics["episode_rewards"][-10:]
                self.metrics["mean_rewards"].append(np.mean(recent))

            if update_count % log_interval == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed
                recent_rewards = self.metrics["episode_rewards"][-10:] if self.metrics["episode_rewards"] else [0]
                print(f"Frames: {frames:>8}/{total_frames} | "
                      f"Loss: {loss:>8.4f} | "
                      f"Reward: {np.mean(recent_rewards):>8.2f} | "
                      f"FPS: {fps:>6.0f}")

        print(f"\nTraining complete in {time.time() - start_time:.1f}s")
        return self.metrics

    def save(self, path):
        """Save model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_dict = {"config": self.config, "metrics": self.metrics}

        if self.algorithm == "ippo":
            save_dict["policies"] = [p.state_dict() for p in self.policies]
            save_dict["critics"] = [c.state_dict() for c in self.critics]
        else:
            save_dict["policy"] = self.policy.state_dict()
            save_dict["critic"] = self.critic.state_dict()

        torch.save(save_dict, path)
        print(f"Model saved: {path}")

    def generate_gif(self, path, num_episodes=1, fps=10):
        """Generate GIF of trained policy using ORIGINAL visualization style."""
        try:
            import imageio
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        except ImportError:
            print("imageio/matplotlib not installed, skipping GIF generation")
            return

        frames_list = []
        for ep in range(num_episodes):
            td = self.env.reset()
            obs = td["observation"].to(self.device)
            done = False
            step = 0

            while not done and step < 300:
                # ORIGINAL v1 STYLE visualization
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xlim(-self.env.arena_radius - self.env.room_radius, self.env.arena_radius + self.env.room_radius)
                ax.set_ylim(-self.env.arena_radius - self.env.room_radius, self.env.arena_radius + self.env.room_radius)
                ax.set_aspect('equal')
                ax.set_title(f"Step {step} | Phase: {self.env.phase}")

                # Arena bounds - ORIGINAL: gray dashed
                ax.add_artist(plt.Circle((0, 0), self.env.arena_radius, fill=False, color="gray", linestyle="--"))

                # Center visual - ORIGINAL style
                if self.env.phase == "spinning":
                    ax.add_artist(plt.Circle((0, 0), self.env.center_radius, fill=True, color="lightblue", alpha=0.3))
                    ax.add_artist(plt.Circle((0, 0), self.env.center_radius, fill=False, color="blue", linestyle="-"))
                else:
                    ax.add_artist(plt.Circle((0, 0), self.env.center_radius, fill=True, color="lightsalmon", alpha=0.3))
                    ax.add_artist(plt.Circle((0, 0), self.env.center_radius, fill=False, color="red", linestyle="-"))

                # Draw rooms - ORIGINAL style
                if hasattr(self.env, "room_positions") and self.env.room_positions is not None:
                    room_positions = self.env.room_positions.cpu().numpy()
                    room_occupancies = self.env.room_occupancy.cpu().numpy() if hasattr(self.env, "room_occupancy") else np.zeros(len(room_positions))

                    for i, (room_pos, occupancy) in enumerate(zip(room_positions, room_occupancies)):
                        fill_ratio = min(occupancy / self.env.room_capacity, 1.0) if hasattr(self.env, "room_capacity") else 0
                        if fill_ratio >= 1.0:
                            color, alpha = "green", 0.4
                        elif fill_ratio > 0:
                            color, alpha = "yellow", 0.3 * fill_ratio
                        else:
                            color, alpha = "green", 0.2

                        ax.add_artist(plt.Circle(room_pos, radius=self.env.room_radius, fill=True, color=color, alpha=alpha))
                        ax.add_artist(plt.Circle(room_pos, radius=self.env.room_radius, fill=False, color="green", linestyle="-"))
                        ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occupancy)}/{self.env.room_capacity}",
                                ha='center', va='center', fontsize=10)

                # Agents - ORIGINAL style
                positions = self.env.agent_positions.detach().cpu().numpy()
                in_center = np.linalg.norm(positions, axis=1) <= self.env.center_radius
                in_room = np.zeros(len(positions), dtype=bool)

                if hasattr(self.env, "room_positions") and self.env.room_positions is not None:
                    room_positions = self.env.room_positions.cpu().numpy()
                    for i, pos in enumerate(positions):
                        distances = np.linalg.norm(room_positions - pos, axis=1)
                        if distances.min() < self.env.room_radius:
                            in_room[i] = True

                outside_all = ~in_center & ~in_room

                # ORIGINAL scatter plot style
                if self.env.phase == "spinning":
                    ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
                    ax.scatter(positions[~in_center, 0], positions[~in_center, 1], c="red", s=80, label="Outside Center")
                else:
                    ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
                    ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
                    ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside All")

                # Agent numbers
                for i, pos in enumerate(positions):
                    ax.text(pos[0] + 0.2, pos[1] + 0.2, str(i), fontsize=8, color='black')

                ax.legend(loc='upper right', fontsize=8)

                # Convert to image
                canvas = FigureCanvas(fig)
                canvas.draw()
                image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                image = image.reshape(canvas.get_width_height()[::-1] + (4,))
                frames_list.append(image[..., :3])
                plt.close(fig)

                # Take action - USE STOCHASTIC SAMPLING for realistic behavior
                with torch.no_grad():
                    if self.algorithm == "ippo":
                        agent_actions = []
                        for i in range(self.config["n_agents"]):
                            mean, std = self.policies[i](obs[i:i+1])
                            mean, std = mean.squeeze(0), std.squeeze(0)
                            # Sample from distribution (stochastic)
                            dist = torch.distributions.Normal(mean, std.clamp(0.1, 1.0))
                            agent_actions.append(dist.sample().clamp(-self.env.max_speed, self.env.max_speed))
                        action = torch.stack(agent_actions)
                    else:
                        mean, std = self.policy(obs.unsqueeze(0))
                        mean, std = mean.squeeze(0), std.squeeze(0)
                        # Sample from distribution (stochastic)
                        dist = torch.distributions.Normal(mean, std.clamp(0.1, 1.0))
                        action = dist.sample().clamp(-self.env.max_speed, self.env.max_speed)

                step_td = TensorDict({"action": action.cpu()}, batch_size=[])
                next_td = self.env._step(step_td)
                done = next_td["done"].any() or next_td["terminated"].any()
                obs = next_td["observation"].to(self.device)
                step += 1

        if frames_list:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            imageio.mimsave(path, frames_list, fps=fps, loop=0)
            print(f"GIF saved: {path}")


def compute_gini(rewards):
    """Compute Gini coefficient."""
    if len(rewards) == 0:
        return 0.0
    rewards = np.array(rewards)
    rewards = rewards - rewards.min() + 1e-8
    sorted_r = np.sort(rewards)
    n = len(sorted_r)
    cum = np.cumsum(sorted_r)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_r)) / (n * cum[-1])) - (n + 1) / n


def plot_comparison(results, metric_name, title, output_path):
    """Create comparison plot."""
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        if metric_name in data and len(data[metric_name]) > 0:
            y = data[metric_name]
            x = np.arange(len(y))
            # Smooth the curve
            window = min(50, len(y) // 10 + 1)
            if len(y) > window:
                y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
                x_smooth = x[:len(y_smooth)]
                plt.plot(x_smooth, y_smooth, label=name, linewidth=2)
            else:
                plt.plot(x, y, label=name, linewidth=2)

    plt.xlabel("Updates")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


def save_summary(results, output_path):
    """Save results summary."""
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "final_reward": np.mean(data["episode_rewards"][-20:]) if data["episode_rewards"] else 0,
            "max_reward": max(data["episode_rewards"]) if data["episode_rewards"] else 0,
            "total_episodes": len(data["episode_rewards"]),
        }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {output_path}")


def run_algorithm_experiments(base_dir, device, total_frames=100000):
    """Run PPO, IPPO, MAPPO comparison."""
    print("\n" + "="*70)
    print("ALGORITHM EXPERIMENTS: PPO vs IPPO vs MAPPO")
    print("="*70)

    output_dir = os.path.join(base_dir, "algorithms")
    results = {}

    for algo in ["ppo", "ippo", "mappo"]:
        algo_dir = os.path.join(output_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)

        config = {
            "algorithm": algo,
            "n_agents": 6,
            "n_rooms": 3,
            "room_capacity": 2,
            "hidden_dim": 128,
            "lr": 3e-4,
            "total_frames": total_frames,
        }

        trainer = ExperimentTrainer(config, device)
        metrics = trainer.train(total_frames)
        trainer.save(os.path.join(algo_dir, "model.pt"))

        # Save metrics
        with open(os.path.join(algo_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Generate GIF
        trainer.generate_gif(os.path.join(algo_dir, "trained_policy.gif"))

        results[algo.upper()] = metrics

    # Create comparison
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    plot_comparison(results, "episode_rewards", "Algorithm Comparison: Episode Rewards",
                   os.path.join(comparison_dir, "reward_comparison.png"))
    plot_comparison(results, "losses", "Algorithm Comparison: Training Loss",
                   os.path.join(comparison_dir, "loss_comparison.png"))
    save_summary(results, os.path.join(comparison_dir, "summary.json"))

    return results


def run_fairness_experiments(base_dir, device, total_frames=100000):
    """Run fairness mode comparison."""
    print("\n" + "="*70)
    print("FAIRNESS EXPERIMENTS: Baseline vs Gini vs Participation")
    print("="*70)

    output_dir = os.path.join(base_dir, "fairness")
    results = {}

    fairness_modes = [
        ("baseline", "none"),
        ("gini", "gini"),
        ("participation", "participation_variance"),
    ]

    for name, mode in fairness_modes:
        mode_dir = os.path.join(output_dir, name)
        os.makedirs(mode_dir, exist_ok=True)

        config = {
            "algorithm": "ppo",
            "n_agents": 6,
            "n_rooms": 3,
            "room_capacity": 2,
            "hidden_dim": 128,
            "lr": 3e-4,
            "fairness_mode": mode,
            "fairness_alpha": 0.5,
        }

        trainer = ExperimentTrainer(config, device)
        metrics = trainer.train(total_frames)
        trainer.save(os.path.join(mode_dir, "model.pt"))

        with open(os.path.join(mode_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        trainer.generate_gif(os.path.join(mode_dir, "trained_policy.gif"))

        results[name.title()] = metrics

    # Comparison
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    plot_comparison(results, "episode_rewards", "Fairness Mode Comparison: Episode Rewards",
                   os.path.join(comparison_dir, "reward_comparison.png"))
    save_summary(results, os.path.join(comparison_dir, "summary.json"))

    return results


def run_communication_experiments(base_dir, device, total_frames=100000):
    """Run communication channel comparison."""
    print("\n" + "="*70)
    print("COMMUNICATION EXPERIMENTS: Baseline vs Discrete vs Continuous")
    print("="*70)

    output_dir = os.path.join(base_dir, "communication")
    results = {}

    # Baseline (no communication)
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    config = {
        "algorithm": "ppo",
        "n_agents": 6,
        "n_rooms": 3,
        "hidden_dim": 128,
        "lr": 3e-4,
    }

    trainer = ExperimentTrainer(config, device)
    metrics = trainer.train(total_frames)
    trainer.save(os.path.join(baseline_dir, "model.pt"))
    with open(os.path.join(baseline_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    trainer.generate_gif(os.path.join(baseline_dir, "trained_policy.gif"))
    results["Baseline"] = metrics

    # Try to run discrete communication
    try:
        from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
        from communication_channel.train_comm import train_discrete_comm

        discrete_dir = os.path.join(output_dir, "discrete")
        os.makedirs(discrete_dir, exist_ok=True)

        print("\nTraining Discrete Communication...")
        discrete_metrics = train_discrete_comm(
            n_agents=6, n_rooms=3, total_frames=total_frames,
            output_dir=discrete_dir, device=device
        )
        results["Discrete"] = discrete_metrics
    except Exception as e:
        print(f"Discrete communication training failed: {e}")
        results["Discrete"] = {"episode_rewards": [], "losses": []}

    # Try continuous communication
    try:
        from communication_channel.envs.continuous_comm_env import MingleEnvWithEmbeddings
        from communication_channel.train_comm import train_continuous_comm

        continuous_dir = os.path.join(output_dir, "continuous")
        os.makedirs(continuous_dir, exist_ok=True)

        print("\nTraining Continuous Communication...")
        continuous_metrics = train_continuous_comm(
            n_agents=6, n_rooms=3, total_frames=total_frames,
            output_dir=continuous_dir, device=device
        )
        results["Continuous"] = continuous_metrics
    except Exception as e:
        print(f"Continuous communication training failed: {e}")
        results["Continuous"] = {"episode_rewards": [], "losses": []}

    # Comparison
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    plot_comparison(results, "episode_rewards", "Communication Comparison: Episode Rewards",
                   os.path.join(comparison_dir, "reward_comparison.png"))
    save_summary(results, os.path.join(comparison_dir, "summary.json"))

    return results


def run_curriculum_experiments(base_dir, device, total_frames=150000):
    """Run curriculum learning: 2 agents, 1 room -> 6 agents, 3 rooms."""
    print("\n" + "="*70)
    print("CURRICULUM LEARNING: Progressive Difficulty")
    print("="*70)

    output_dir = os.path.join(base_dir, "curriculum", "progressive")
    os.makedirs(output_dir, exist_ok=True)

    # Curriculum: progressively harder environments
    # Each stage gets more frames to handle increased complexity
    stages = [
        {"n_agents": 2, "n_rooms": 1, "frames": total_frames // 5},
        {"n_agents": 3, "n_rooms": 2, "frames": total_frames // 5},
        {"n_agents": 4, "n_rooms": 2, "frames": total_frames // 5},
        {"n_agents": 5, "n_rooms": 3, "frames": total_frames // 5},
        {"n_agents": 6, "n_rooms": 3, "frames": total_frames // 5},
    ]

    all_metrics = {"episode_rewards": [], "losses": [], "stages": []}

    # Track previous observation dimension for weight transfer
    prev_obs_dim = None
    policy_state = None
    critic_state = None

    for i, stage in enumerate(stages):
        print(f"\n--- Stage {i+1}: {stage['n_agents']} agents, {stage['n_rooms']} rooms ---")

        stage_dir = os.path.join(output_dir, f"stage_{i+1}")
        os.makedirs(stage_dir, exist_ok=True)

        config = {
            "algorithm": "ppo",
            "n_agents": stage["n_agents"],
            "n_rooms": stage["n_rooms"],
            "hidden_dim": 128,
            "lr": 3e-4,
        }

        trainer = ExperimentTrainer(config, device)

        # Get current observation dimension
        current_obs_dim = trainer.obs_dim

        # Transfer weights only if dimensions match exactly
        if policy_state is not None and prev_obs_dim == current_obs_dim:
            try:
                trainer.policy.load_state_dict(policy_state)
                trainer.critic.load_state_dict(critic_state)
                print("Transferred weights from previous stage (same dimensions)")
            except Exception as e:
                print(f"Could not transfer weights: {e}")
        elif policy_state is not None:
            print(f"Skipping weight transfer: obs_dim changed from {prev_obs_dim} to {current_obs_dim}")

        # Update previous obs_dim
        prev_obs_dim = current_obs_dim

        metrics = trainer.train(stage["frames"])
        trainer.save(os.path.join(stage_dir, "model.pt"))

        with open(os.path.join(stage_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        trainer.generate_gif(os.path.join(stage_dir, "trained_policy.gif"))

        # Save state for transfer
        policy_state = trainer.policy.state_dict()
        critic_state = trainer.critic.state_dict()

        all_metrics["episode_rewards"].extend(metrics["episode_rewards"])
        all_metrics["losses"].extend(metrics["losses"])
        all_metrics["stages"].append({
            "stage": i + 1,
            "n_agents": stage["n_agents"],
            "n_rooms": stage["n_rooms"],
            "final_reward": np.mean(metrics["episode_rewards"][-20:]) if metrics["episode_rewards"] else 0
        })

    # Save overall metrics
    with open(os.path.join(output_dir, "curriculum_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Plot curriculum learning curve
    plt.figure(figsize=(12, 6))
    rewards = all_metrics["episode_rewards"]
    if rewards:
        window = min(50, len(rewards) // 10 + 1)
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, linewidth=2)
        else:
            plt.plot(rewards, linewidth=2)

        # Mark stage transitions
        total = 0
        for stage in all_metrics["stages"][:-1]:
            idx = stage.get("stage", 1)
            if idx < len(stages):
                total += stages[idx-1]["frames"] // 1024
                plt.axvline(x=min(total, len(rewards)-1), color='r', linestyle='--', alpha=0.5)

    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.title("Curriculum Learning: Progressive Difficulty")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "curriculum_progress.png"), dpi=150)
    plt.close()

    return all_metrics


def run_combined_experiments(base_dir, device, total_frames=150000):
    """Run with ALL features combined."""
    print("\n" + "="*70)
    print("ALL COMBINED: MAPPO + Fairness + Communication")
    print("="*70)

    output_dir = os.path.join(base_dir, "all_combined")
    os.makedirs(output_dir, exist_ok=True)

    # MAPPO with Gini fairness (best combo from individual experiments)
    config = {
        "algorithm": "mappo",
        "n_agents": 6,
        "n_rooms": 3,
        "room_capacity": 2,
        "hidden_dim": 128,
        "lr": 3e-4,
        "fairness_mode": "gini",
        "fairness_alpha": 0.5,
    }

    trainer = ExperimentTrainer(config, device)
    metrics = trainer.train(total_frames)
    trainer.save(os.path.join(output_dir, "model.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    trainer.generate_gif(os.path.join(output_dir, "trained_policy.gif"))

    # Create summary plot
    plt.figure(figsize=(10, 6))
    rewards = metrics["episode_rewards"]
    if rewards:
        window = min(50, len(rewards) // 10 + 1)
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, linewidth=2, label="MAPPO + Gini Fairness")
        else:
            plt.plot(rewards, linewidth=2, label="MAPPO + Gini Fairness")

    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.title("All Features Combined: MAPPO + Fairness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "combined_training.png"), dpi=150)
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--algorithms", action="store_true", help="Run algorithm comparison")
    parser.add_argument("--communication", action="store_true", help="Run communication comparison")
    parser.add_argument("--fairness", action="store_true", help="Run fairness comparison")
    parser.add_argument("--curriculum", action="store_true", help="Run curriculum learning")
    parser.add_argument("--combined", action="store_true", help="Run combined features")
    parser.add_argument("--frames", type=int, default=100000, help="Total frames per experiment")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # If no specific experiment selected, run all
    if not any([args.algorithms, args.communication, args.fairness, args.curriculum, args.combined]):
        args.all = True

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Base output directory
    base_dir = "contribution_tests_and_comparisions"
    os.makedirs(base_dir, exist_ok=True)

    start_time = time.time()

    if args.all or args.algorithms:
        run_algorithm_experiments(base_dir, device, args.frames)

    if args.all or args.fairness:
        run_fairness_experiments(base_dir, device, args.frames)

    if args.all or args.communication:
        run_communication_experiments(base_dir, device, args.frames)

    if args.all or args.curriculum:
        run_curriculum_experiments(base_dir, device, int(args.frames * 1.5))

    if args.all or args.combined:
        run_combined_experiments(base_dir, device, int(args.frames * 1.5))

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {base_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
