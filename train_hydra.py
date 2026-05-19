"""
Hydra-based Training Script for Cooperative Mingle

Usage:
    # Single experiment
    python train_hydra.py
    python train_hydra.py algorithm=mappo
    python train_hydra.py algorithm=mappo fairness=gini communication=discrete

    # Sweeps (multiple experiments)
    python train_hydra.py --multirun algorithm=ppo,ippo,mappo
    python train_hydra.py --multirun communication=none,discrete,continuous
    python train_hydra.py --multirun fairness=none,gini,participation

    # Full stack
    python train_hydra.py algorithm=mappo communication=discrete fairness=gini curriculum=progressive
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

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    StayInRoomReward,
    GetToRoomReward,
    EfficiencyReward,
    FairnessReward,
    MultiObjectiveReward,
)


class BasePolicy(nn.Module):
    """Policy network with agent embeddings."""
    def __init__(self, obs_dim, action_dim=2, hidden_dim=128, n_agents=6):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
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
        was_flat = False
        original_shape = obs.shape

        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                was_flat = True
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
            else:
                # Fallback for flat observations
                was_flat = True
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
        agent_ids = torch.arange(n_agents, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        agent_embeds = self.agent_embedding(agent_ids)
        obs_with_id = torch.cat([obs, agent_embeds], dim=-1)

        room_dir = obs[:, :, 5:7]
        obs_flat = obs_with_id.view(-1, obs_dim + self.agent_embed_dim)
        hidden = self.net(obs_flat)
        adjustment = torch.tanh(self.action_head(hidden))
        adjustment = adjustment.view(batch_size, n_agents, -1)

        loc = room_dir * 0.25 + adjustment * 0.05
        scale = self.log_std.exp().expand_as(loc)

        if was_flat:
            return loc.view(-1, self.action_dim), scale.view(-1, self.action_dim)

        return loc.squeeze(0), scale.squeeze(0)


class Critic(nn.Module):
    """Value network."""
    def __init__(self, obs_dim, hidden_dim=128, centralized=False, n_agents=6):
        super().__init__()
        self.centralized = centralized
        input_dim = obs_dim * n_agents if centralized else obs_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if not centralized else n_agents),
        )

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        batch_size, n_agents, obs_dim = obs.shape

        if self.centralized:
            obs_flat = obs.view(batch_size, -1)
            values = self.net(obs_flat)
            return values.squeeze(0)
        else:
            obs_flat = obs.view(-1, obs_dim)
            values = self.net(obs_flat)
            return values.view(batch_size, n_agents, 1).squeeze(0)


class ObservationNormalizer:
    """Running observation normalizer."""
    def __init__(self, obs_dim, device):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 0
        self.device = device

    def update(self, obs):
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
        if self.count == 0:
            return obs
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


def get_reward_modules(cfg):
    """Create reward modules from config."""
    if cfg.fairness.mode == "pareto":
        fairness_metric = cfg.fairness.get("fairness_metric", "gini")
        occupancy_weight = cfg.fairness.get("efficiency_occupancy_weight", 1.0)
        success_weight = cfg.fairness.get("efficiency_success_weight", 1.0)

        efficiency = EfficiencyReward(
            phase_mode="claiming",
            occupancy_weight=occupancy_weight,
            success_weight=success_weight,
        )
        fairness = FairnessReward(
            phase_mode="claiming",
            fairness_metric=fairness_metric,
        )
        scalarized = MultiObjectiveReward(
            alpha=cfg.fairness.alpha,
            efficiency_module=efficiency,
            fairness_module=fairness,
            phase_mode="claiming",
        )
        scalarized._activate()
        return [scalarized]

    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=5.0, phase_mode="claiming"),
    ]
    for m in modules:
        m._activate()
    return modules


def compute_gini(values):
    """Compute Gini coefficient for fairness measurement."""
    values = np.array(values).flatten()
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.abs(values)
    values = np.sort(values)
    n = len(values)
    return (2 * np.sum(np.arange(1, n + 1) * values) / (n * np.sum(values))) - (n + 1) / n


def compute_jain(values):
    """Compute Jain's fairness index for a vector."""
    values = np.array(values).flatten().astype(float)
    if len(values) == 0:
        return 0.0
    numerator = np.sum(values) ** 2
    denominator = len(values) * np.sum(values ** 2)
    if denominator == 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def compute_episode_efficiency(env: MingleEnv) -> float:
    """Compute a scalar efficiency score from the final episode state."""
    if env.room_positions is None:
        return 0.0

    room_dists = torch.cdist(env.agent_positions, env.room_positions)
    in_room = room_dists < env.room_radius

    assignments = torch.full((env.n_agents,), -1, dtype=torch.long, device=env.device)
    for i in range(env.n_agents):
        if in_room[i].any():
            assignments[i] = in_room[i].nonzero(as_tuple=True)[0][0]

    room_occupancy = torch.bincount(assignments[assignments >= 0], minlength=env.n_rooms)
    agents_in_valid_rooms = 0
    for i in range(env.n_agents):
        room_idx = assignments[i].item()
        if room_idx >= 0 and room_occupancy[room_idx] <= env.room_capacity:
            agents_in_valid_rooms += 1

    occupancy_rate = agents_in_valid_rooms / max(env.n_agents, 1)
    success = 1.0 if agents_in_valid_rooms == env.n_agents else 0.0
    return float(0.5 * occupancy_rate + 0.5 * success)


def get_device(cfg):
    """Get device from config."""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def train(cfg: DictConfig):
    """Main training function."""
    device = get_device(cfg)
    print(f"\n{'='*60}")
    print(f"Training: {cfg.experiment_name}")
    print(f"Algorithm: {cfg.algorithm.name}")
    print(f"Communication: {cfg.communication.name}")
    print(f"Fairness: {cfg.fairness.mode}")
    print(f"Curriculum: {cfg.curriculum.name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create environment
    env = MingleEnv(
        n_agents=cfg.env.n_agents,
        n_rooms=cfg.env.n_rooms,
        room_capacity=cfg.env.room_capacity,
        max_steps=cfg.env.max_steps,
        phase_mode=cfg.env.phase_mode,
        reward_modules=get_reward_modules(cfg),
    )

    # Enable communication if configured
    env.enable_communication = cfg.communication.enabled

    # Calculate observation dimension
    obs_dim = 14 + cfg.communication.obs_dim_addition
    n_agents = cfg.env.n_agents

    # Create networks
    policy = BasePolicy(obs_dim, n_agents=n_agents, hidden_dim=cfg.algorithm.hidden_dim).to(device)
    centralized = cfg.algorithm.get("centralized_critic", False)
    critic = Critic(obs_dim, hidden_dim=cfg.algorithm.hidden_dim, centralized=centralized, n_agents=n_agents).to(device)
    obs_normalizer = ObservationNormalizer(obs_dim, device)

    # Optimizers
    policy_optim = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr, weight_decay=0.01)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=cfg.train.lr, weight_decay=0.01)

    # Metrics
    metrics = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "episode_rewards": [],
        "gini_coefficients": [],
        "episode_fairness": [],
        "episode_efficiency": [],
        "losses": [],
    }

    # Training loop
    frames = 0
    total_frames = cfg.train.total_frames
    start_time = time.time()

    # Curriculum learning stages
    if cfg.curriculum.enabled:
        stages = cfg.curriculum.stages
        current_stage_idx = 0
    else:
        stages = None
        current_stage_idx = None

    while frames < total_frames:
        # Update curriculum stage if enabled
        if stages:
            stage_frames = sum(s.frames for s in stages[:current_stage_idx+1])
            if frames >= stage_frames and current_stage_idx < len(stages) - 1:
                current_stage_idx += 1
                stage = stages[current_stage_idx]
                env.room_capacity = stage.room_capacity
                print(f"\n[Curriculum] Stage {current_stage_idx+1}: capacity={stage.room_capacity}, penalty={stage.overfill_penalty}")

        # Collect experience
        observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        td = env.reset()
        obs_raw = td["observation"].to(device)

        episode_reward = 0
        agent_episode_rewards = torch.zeros(n_agents)
        epsilon = max(0.1, 0.5 * (1 - frames / total_frames))

        for _ in range(cfg.train.frames_per_batch):
            obs_normalizer.update(obs_raw.unsqueeze(0))
            obs = obs_normalizer.normalize(obs_raw)

            with torch.no_grad():
                loc, scale = policy(obs.unsqueeze(0))
                scale = scale.clamp(0.2, 1.0)
                dist = torch.distributions.Normal(loc, scale)
                action = dist.sample()

                if torch.rand(1).item() < epsilon:
                    action = torch.rand_like(action) * 0.6 - 0.3

                action = action.clamp(-0.3, 0.3)
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = critic(obs.unsqueeze(0))

            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            reward = next_td["reward"].to(device)

            # Apply fairness penalty if configured
            if cfg.fairness.mode == "gini":
                agent_rewards = reward.squeeze()
                gini = compute_gini(agent_rewards.cpu().numpy())
                reward = reward - cfg.fairness.alpha * gini

            done = next_td["done"].any() or next_td["terminated"].any()

            observations.append(obs)
            actions.append(action)
            rewards.append(reward.squeeze())
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.squeeze() if value.dim() > 1 else value)

            episode_reward += reward.sum().item()
            agent_episode_rewards += reward.squeeze().cpu()

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["gini_coefficients"].append(compute_gini(agent_episode_rewards.numpy()))
                metrics["episode_fairness"].append(compute_jain(agent_episode_rewards.numpy()))
                metrics["episode_efficiency"].append(compute_episode_efficiency(env))
                episode_reward = 0
                agent_episode_rewards = torch.zeros(n_agents)
                td = env.reset()
                obs_raw = td["observation"].to(device)
            else:
                obs_raw = next_td["observation"].to(device)

        frames += cfg.train.frames_per_batch * n_agents

        # PPO update
        obs_tensor = torch.stack(observations)
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.stack(rewards)
        dones_tensor = torch.tensor(dones, device=device)
        old_log_probs = torch.stack(log_probs)
        values_tensor = torch.stack(values)

        with torch.no_grad():
            obs = obs_normalizer.normalize(obs_raw)
            next_value = critic(obs.unsqueeze(0)).squeeze()

        # GAE
        advantages = torch.zeros_like(rewards_tensor)
        gae = torch.zeros(n_agents, device=device)

        for t in reversed(range(len(rewards_tensor))):
            if t == len(rewards_tensor) - 1:
                next_val = next_value
            else:
                next_val = values_tensor[t + 1]
            done_mask = 1 - dones_tensor[t].float()
            delta = rewards_tensor[t] + cfg.algorithm.gamma * next_val * done_mask - values_tensor[t]
            gae = delta + cfg.algorithm.gamma * cfg.algorithm.lmbda * done_mask * gae
            advantages[t] = gae

        returns = advantages + values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten for training
        T = len(rewards_tensor)
        obs_flat = obs_tensor.view(T * n_agents, -1)
        actions_flat = actions_tensor.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        total_loss = 0
        for _ in range(cfg.train.num_epochs):
            loc, scale = policy(obs_flat)
            scale = scale.clamp(0.1, 1.0)
            dist = torch.distributions.Normal(loc, scale)
            new_log_probs = dist.log_prob(actions_flat).sum(dim=-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - cfg.algorithm.clip_epsilon, 1 + cfg.algorithm.clip_epsilon) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = critic(obs_flat.view(T, n_agents, -1)).view(-1)
            value_loss = nn.functional.mse_loss(new_values, returns_flat)

            loss = policy_loss + cfg.algorithm.value_coef * value_loss - cfg.algorithm.entropy_coef * entropy

            policy_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.train.max_grad_norm)
            policy_optim.step()
            critic_optim.step()
            total_loss += loss.item()

        metrics["losses"].append(total_loss / cfg.train.num_epochs)

        # Logging
        if len(metrics["episode_rewards"]) > 0 and frames % cfg.train.log_interval < cfg.train.frames_per_batch * n_agents:
            recent_rewards = metrics["episode_rewards"][-10:]
            recent_gini = metrics["gini_coefficients"][-10:]
            elapsed = time.time() - start_time
            print(f"Frames: {frames:8d}/{total_frames} | "
                  f"Reward: {np.mean(recent_rewards):8.2f} | "
                  f"Gini: {np.mean(recent_gini):.3f} | "
                  f"Time: {elapsed:.1f}s")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    if cfg.train.save_model:
        torch.save({
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
        }, output_dir / "model.pt")
        print(f"Model saved: {output_dir / 'model.pt'}")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")

    # Generate GIF
    generate_gif(env, policy, obs_normalizer, output_dir / "trained_policy.gif", cfg.experiment_name, n_agents, device)

    # Final summary
    final_reward = np.mean(metrics["episode_rewards"][-20:]) if metrics["episode_rewards"] else 0
    final_gini = np.mean(metrics["gini_coefficients"][-20:]) if metrics["gini_coefficients"] else 0
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Reward: {final_reward:.2f}")
    print(f"Final Gini: {final_gini:.3f}")
    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"{'='*60}")

    return final_reward, final_gini


def generate_gif(env, policy, obs_normalizer, output_path, title, n_agents, device):
    """Generate visualization GIF."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(device)

        for step in range(200):
            obs = obs_normalizer.normalize(obs_raw)
            with torch.no_grad():
                loc, scale = policy(obs.unsqueeze(0))
                noise = torch.randn_like(loc) * scale * 0.3
                actions = (loc + noise).clamp(-0.3, 0.3)

            step_td = TensorDict({"action": actions.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')
            ax.set_title(f"{title} | Step {step}", fontsize=14)

            ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))

            for i, (room_pos, occ) in enumerate(zip(env.room_positions.cpu().numpy(), env.room_occupancy.cpu().numpy())):
                color = "green" if occ >= env.room_capacity else "yellow" if occ > 0 else "lightgreen"
                ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=True, color=color, alpha=0.3))
                ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=False, color="green"))
                ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occ)}/{env.room_capacity}",
                        ha='center', va='center', fontsize=10, fontweight='bold')

            positions = env.agent_positions.cpu().numpy()
            colors_list = plt.cm.tab10(np.linspace(0, 1, n_agents))

            for i, pos in enumerate(positions):
                if env.enable_communication and hasattr(env, 'is_leader') and env.is_leader[i]:
                    marker = 's'
                    label = f"L{i}"
                else:
                    marker = 'o'
                    label = f"A{i}"
                ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, marker=marker,
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(label, (pos[0], pos[1] + 0.7), ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            gif_frames.append(frame[..., :3])
            plt.close(fig)

            if next_td["done"].any() or next_td["terminated"].any():
                break
            obs_raw = next_td["observation"].to(device)

        imageio.mimsave(str(output_path), gif_frames, fps=10, loop=0)
        print(f"GIF saved: {output_path}")
    except Exception as e:
        print(f"GIF generation failed: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
