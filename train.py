"""
Hydra-based Training Script for Cooperative Mingle

Usage:
    # Train with default config (PPO, no fairness, 4 agents)
    python train.py

    # Train with MAPPO
    python train.py algorithm=mappo

    # Train with Gini fairness
    python train.py fairness=gini

    # Train with 6 agents, 3 rooms
    python train.py env.n_agents=6 env.n_rooms=3

    # Run multiple experiments (sweep)
    python train.py --multirun algorithm=ppo,ippo,mappo seed=0,1,2

    # Full experiment example
    python train.py algorithm=mappo fairness=gini env.n_agents=6 train.total_frames=200000
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform


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


class PPOTrainer:
    """PPO Trainer with Hydra config support."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Create environment
        self.env = MingleEnv(
            n_agents=cfg.env.n_agents,
            n_rooms=cfg.env.n_rooms,
            room_capacity=cfg.env.room_capacity,
            arena_radius=cfg.env.arena_radius,
            center_radius=cfg.env.center_radius,
            room_radius=cfg.env.room_radius,
            max_steps=cfg.env.max_steps,
            phase_mode=cfg.env.phase_mode,
        )

        # Create fairness transform if enabled
        self.fairness_transform = None
        if cfg.fairness.enabled:
            self.fairness_transform = make_fairness_reward_transform({
                "mode": cfg.fairness.mode,
                "alpha": cfg.fairness.alpha,
            })

        # Get dimensions
        obs_dim = self.env.observation_spec["observation"].shape[-1]
        action_dim = 2

        # Create networks
        self.policy = PolicyNetwork(
            obs_dim, action_dim,
            hidden_dim=cfg.algorithm.hidden_dim,
            num_layers=cfg.algorithm.num_layers
        ).to(device)

        self.critic = CriticNetwork(
            obs_dim,
            hidden_dim=cfg.algorithm.hidden_dim,
            num_layers=cfg.algorithm.num_layers
        ).to(device)

        # Optimizers
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.train.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.lr)

        # Metrics
        self.metrics = {
            "episode_rewards": [],
            "losses": [],
            "policy_losses": [],
            "value_losses": [],
        }

    def collect_rollout(self, num_steps):
        """Collect experience from environment."""
        observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        td = self.env.reset()
        obs = td["observation"].to(self.device)

        for _ in range(num_steps):
            with torch.no_grad():
                mean, std = self.policy(obs.unsqueeze(0))
                mean, std = mean.squeeze(0), std.squeeze(0)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().clamp(-self.env.max_speed, self.env.max_speed)
                log_prob = dist.log_prob(action).sum(dim=-1)
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

            if done:
                self.metrics["episode_rewards"].append(reward.sum().item())
                td = self.env.reset()
                obs = td["observation"].to(self.device)
                if self.fairness_transform:
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

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        cfg = self.cfg.algorithm
        T, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(n_agents, device=rewards.device)

        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else torch.zeros(n_agents, device=rewards.device)
            mask = 1 - dones[t].float()
            delta = rewards[t] + cfg.gamma * next_val * mask - values[t]
            advantages[t] = delta + cfg.gamma * cfg.lmbda * mask * last_adv
            last_adv = advantages[t]

        return advantages, advantages + values

    def update(self, rollout):
        """PPO update step."""
        cfg = self.cfg
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

        total_loss, policy_loss_sum, value_loss_sum = 0, 0, 0

        for _ in range(cfg.train.num_epochs):
            mean, std = self.policy(obs_flat)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_flat).sum(dim=-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - cfg.algorithm.clip_epsilon,
                               1 + cfg.algorithm.clip_epsilon) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(obs_flat).squeeze(-1)
            value_loss = nn.functional.mse_loss(new_values, returns_flat)

            loss = policy_loss + cfg.algorithm.value_coef * value_loss - cfg.algorithm.entropy_coef * entropy

            self.policy_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.train.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.train.max_grad_norm)
            self.policy_optim.step()
            self.critic_optim.step()

            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()

        n_epochs = cfg.train.num_epochs
        self.metrics["losses"].append(total_loss / n_epochs)
        self.metrics["policy_losses"].append(policy_loss_sum / n_epochs)
        self.metrics["value_losses"].append(value_loss_sum / n_epochs)

        return total_loss / n_epochs

    def train(self):
        """Main training loop."""
        cfg = self.cfg
        frames = 0
        update_count = 0
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training: {cfg.algorithm.name.upper()}")
        print(f"Environment: {cfg.env.n_agents} agents, {cfg.env.n_rooms} rooms")
        print(f"Fairness: {cfg.fairness.mode}")
        print(f"Total frames: {cfg.train.total_frames}")
        print(f"{'='*60}\n")

        while frames < cfg.train.total_frames:
            rollout = self.collect_rollout(cfg.train.frames_per_batch // self.env.n_agents)
            frames += cfg.train.frames_per_batch
            loss = self.update(rollout)
            update_count += 1

            if update_count % cfg.train.log_interval == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed
                recent_rewards = self.metrics["episode_rewards"][-10:] if self.metrics["episode_rewards"] else [0]
                print(f"Frames: {frames:>8}/{cfg.train.total_frames} | "
                      f"Loss: {loss:>8.4f} | "
                      f"Reward: {np.mean(recent_rewards):>8.2f} | "
                      f"FPS: {fps:>6.0f}")

        print(f"\nTraining complete in {time.time() - start_time:.1f}s")
        return self.metrics

    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "config": OmegaConf.to_container(self.cfg),
            "metrics": self.metrics,
        }, path)
        print(f"Model saved: {path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Device: {device}")

    # Create trainer
    trainer = PPOTrainer(cfg, device)

    # Train
    metrics = trainer.train()

    # Save model
    model_path = os.path.join(os.getcwd(), "model.pt")
    trainer.save(model_path)

    # Save metrics
    metrics_path = os.path.join(os.getcwd(), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Return final reward for Hydra sweeps
    final_reward = np.mean(metrics["episode_rewards"][-20:]) if metrics["episode_rewards"] else 0
    return final_reward


if __name__ == "__main__":
    main()
