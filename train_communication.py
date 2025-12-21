"""
Communication Comparison: Baseline vs Discrete vs Continuous Communication

Discrete Communication (10 pts):
- n/2 agents are "leaders" who broadcast "follow me"
- n/2 agents are "followers" who pick a leader
- If a leader has >1 follower, they broadcast "full"
- Rejected followers must find another leader

Continuous Communication (5 pts):
- Each agent outputs a learned embedding vector (8 dimensions)
- Embeddings are shared with all other agents
- Agents use mean of other agents' embeddings as additional observation
- Embeddings are learned end-to-end through backpropagation

Comparison baseline (5 pts):
- Baseline (no communication) vs Discrete vs Continuous
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
BASE_DIR = Path("contribution_tests_and_comparisions/communication")


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


class CommunicationPolicy(nn.Module):
    """Policy that uses discrete communication features."""
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
        original_shape = obs.shape
        was_flat = False

        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                was_flat = True
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
            else:
                # Unknown shape - handle as flat
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

        # Room direction is at indices 5-6 in base obs
        room_dir = obs[:, :, 5:7]

        obs_flat = obs_with_id.view(-1, obs_dim + self.agent_embed_dim)
        hidden = self.net(obs_flat)

        adjustment = torch.tanh(self.action_head(hidden))
        adjustment = adjustment.view(batch_size, n_agents, -1)

        # Goal-conditioned action
        loc = room_dir * 0.25 + adjustment * 0.05
        scale = self.log_std.exp().expand_as(loc)

        if was_flat:
            return loc.view(-1, self.action_dim), scale.view(-1, self.action_dim)

        return loc.squeeze(0), scale.squeeze(0)


class ContinuousCommPolicy(nn.Module):
    """Policy with continuous learned communication embeddings."""
    def __init__(self, obs_dim, action_dim=2, hidden_dim=128, n_agents=6, comm_dim=8):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.comm_dim = comm_dim

        self.agent_embed_dim = 8
        self.agent_embedding = nn.Embedding(n_agents, self.agent_embed_dim)

        # Communication encoder: generates message embedding from observation
        self.comm_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, comm_dim),
            nn.Tanh(),  # Bound embeddings to [-1, 1]
        )

        # Main policy network: takes obs + agent_embed + received_messages
        # received_messages = mean of other agents' embeddings = comm_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + self.agent_embed_dim + comm_dim, hidden_dim),
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

    def encode_messages(self, obs):
        """Generate communication embeddings for all agents."""
        # obs: (n_agents, obs_dim) or (batch, n_agents, obs_dim)
        if obs.dim() == 2:
            return self.comm_encoder(obs)  # (n_agents, comm_dim)
        else:
            batch_size, n_agents, obs_dim = obs.shape
            obs_flat = obs.view(-1, obs_dim)
            msgs = self.comm_encoder(obs_flat)
            return msgs.view(batch_size, n_agents, -1)

    def aggregate_messages(self, messages, agent_idx=None):
        """Aggregate messages from other agents (mean pooling excluding self)."""
        # messages: (n_agents, comm_dim) or (batch, n_agents, comm_dim)
        if messages.dim() == 2:
            n_agents = messages.shape[0]
            aggregated = []
            for i in range(n_agents):
                # Mean of all OTHER agents' messages
                mask = torch.ones(n_agents, dtype=torch.bool, device=messages.device)
                mask[i] = False
                other_msgs = messages[mask]  # (n_agents-1, comm_dim)
                aggregated.append(other_msgs.mean(dim=0))
            return torch.stack(aggregated)  # (n_agents, comm_dim)
        else:
            batch_size, n_agents, comm_dim = messages.shape
            aggregated = []
            for i in range(n_agents):
                mask = torch.ones(n_agents, dtype=torch.bool, device=messages.device)
                mask[i] = False
                other_msgs = messages[:, mask, :]  # (batch, n_agents-1, comm_dim)
                aggregated.append(other_msgs.mean(dim=1))
            return torch.stack(aggregated, dim=1)  # (batch, n_agents, comm_dim)

    def forward(self, obs, return_messages=False):
        """Forward pass with continuous communication."""
        was_flat = False

        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                was_flat = True
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
            else:
                # Fallback for unknown shapes
                was_flat = True
                n_samples = obs.shape[0]
                agent_ids = torch.arange(n_samples, device=obs.device) % self.n_agents
                agent_embeds = self.agent_embedding(agent_ids)
                # No communication aggregation possible, use zeros
                zero_msgs = torch.zeros(n_samples, self.comm_dim, device=obs.device)
                obs_with_comm = torch.cat([obs, agent_embeds, zero_msgs], dim=-1)
                hidden = self.net(obs_with_comm)
                adjustment = torch.tanh(self.action_head(hidden))
                room_dir = obs[:, 5:7]
                loc = room_dir * 0.25 + adjustment * 0.05
                scale = self.log_std.exp().expand_as(loc)
                if return_messages:
                    return loc, scale, zero_msgs
                return loc, scale

        batch_size, n_agents, obs_dim = obs.shape

        # Generate messages from all agents
        messages = self.encode_messages(obs)  # (batch, n_agents, comm_dim)

        # Aggregate messages for each agent (mean of others)
        received_msgs = self.aggregate_messages(messages)  # (batch, n_agents, comm_dim)

        # Agent embeddings
        agent_ids = torch.arange(n_agents, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        agent_embeds = self.agent_embedding(agent_ids)

        # Concatenate obs + agent_embed + received_messages
        obs_with_comm = torch.cat([obs, agent_embeds, received_msgs], dim=-1)

        # Room direction for goal-conditioned action
        room_dir = obs[:, :, 5:7]

        # Forward through network
        obs_flat = obs_with_comm.view(-1, obs_dim + self.agent_embed_dim + self.comm_dim)
        hidden = self.net(obs_flat)

        adjustment = torch.tanh(self.action_head(hidden))
        adjustment = adjustment.view(batch_size, n_agents, -1)

        loc = room_dir * 0.25 + adjustment * 0.05
        scale = self.log_std.exp().expand_as(loc)

        if was_flat:
            loc = loc.view(-1, self.action_dim)
            scale = scale.view(-1, self.action_dim)
            if return_messages:
                return loc, scale, messages.view(-1, self.comm_dim)
            return loc, scale

        if return_messages:
            return loc.squeeze(0), scale.squeeze(0), messages.squeeze(0)
        return loc.squeeze(0), scale.squeeze(0)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0)
        nn.init.normal_(self.agent_embedding.weight, mean=0, std=0.5)


class Critic(nn.Module):
    """Shared critic."""
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
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        batch_size, n_agents, obs_dim = obs.shape
        obs_flat = obs.view(-1, obs_dim)
        values = self.net(obs_flat)
        return values.view(batch_size, n_agents, 1).squeeze(0)


def get_reward_modules():
    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=5.0, phase_mode="claiming"),
    ]
    for m in modules:
        m._activate()
    return modules


def compute_gini(values):
    values = np.array(values).flatten()
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.abs(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


def train_communication(enable_comm, config, name):
    """Train with or without communication."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Communication: {'ENABLED' if enable_comm else 'DISABLED'}")
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
    env.enable_communication = enable_comm

    obs_dim = 18 if enable_comm else 14
    n_agents = config["n_agents"]

    policy = CommunicationPolicy(obs_dim, n_agents=n_agents).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    obs_normalizer = ObservationNormalizer(obs_dim, DEVICE)

    policy_optim = torch.optim.AdamW(policy.parameters(), lr=config["lr"], weight_decay=0.01)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config["lr"], weight_decay=0.01)

    metrics = {
        "name": name,
        "communication": enable_comm,
        "episode_rewards": [],
        "gini_coefficients": [],
        "losses": [],
    }

    frames = 0
    start_time = time.time()

    while frames < config["total_frames"]:
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
        epsilon = max(0.1, 0.5 * (1 - frames / config["total_frames"]))

        for _ in range(config["frames_per_batch"]):
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

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["gini_coefficients"].append(compute_gini(agent_episode_rewards.numpy()))
                episode_reward = 0
                agent_episode_rewards = torch.zeros(n_agents)
                td = env.reset()
                obs_raw = td["observation"].to(DEVICE)
            else:
                obs_raw = next_td["observation"].to(DEVICE)

        frames += config["frames_per_batch"] * n_agents

        # PPO update
        obs_tensor = torch.stack(observations)
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.stack(rewards)
        dones_tensor = torch.tensor(dones, device=DEVICE)
        old_log_probs = torch.stack(log_probs)
        values_tensor = torch.stack(values)

        with torch.no_grad():
            obs = obs_normalizer.normalize(obs_raw)
            next_value = critic(obs.unsqueeze(0)).squeeze()

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

        T = len(rewards_tensor)
        obs_flat = obs_tensor.view(T * n_agents, -1)
        actions_flat = actions_tensor.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        total_loss = 0
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

            loss = policy_loss + 0.5 * value_loss - config["entropy_coef"] * entropy

            policy_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            policy_optim.step()
            critic_optim.step()
            total_loss += loss.item()

        metrics["losses"].append(total_loss / config["num_epochs"])

        if len(metrics["episode_rewards"]) > 0 and frames % 20000 < config["frames_per_batch"] * n_agents:
            recent_rewards = metrics["episode_rewards"][-10:]
            recent_gini = metrics["gini_coefficients"][-10:]
            print(f"Frames: {frames:8d}/{config['total_frames']} | "
                  f"Reward: {np.mean(recent_rewards):8.2f} | "
                  f"Gini: {np.mean(recent_gini):.3f} | "
                  f"Eps: {epsilon:.2f}")

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")

    # Save results
    output_dir = BASE_DIR / name.lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
    }, output_dir / "model.pt")

    # Generate GIF
    generate_gif(env, policy, obs_normalizer, output_dir / "trained_policy.gif", name, n_agents)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved: {output_dir / 'model.pt'}")
    print(f"Metrics saved: {output_dir / 'metrics.json'}")

    return metrics


def generate_gif(env, policy, obs_normalizer, output_path, title, n_agents):
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        for step in range(300):
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

            if env.room_positions is not None:
                for i, (room_pos, occ) in enumerate(zip(env.room_positions.cpu().numpy(), env.room_occupancy.cpu().numpy())):
                    color = "green" if occ >= env.room_capacity else "yellow" if occ > 0 else "lightgreen"
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=True, color=color, alpha=0.3))
                    ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=False, color="green"))
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occ)}/{env.room_capacity}",
                            ha='center', va='center', fontsize=10, fontweight='bold')

            positions = env.agent_positions.cpu().numpy()
            colors_list = plt.cm.tab10(np.linspace(0, 1, n_agents))

            for i, pos in enumerate(positions):
                # Show leaders vs followers if communication enabled
                if env.enable_communication and env.is_leader[i]:
                    marker = 's'  # Square for leaders
                    label = f"L{i}"
                else:
                    marker = 'o'  # Circle for followers
                    label = f"A{i}"

                ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, marker=marker,
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(label, (pos[0], pos[1] + 0.7), ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Add legend for communication mode
            if env.enable_communication:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                           markersize=12, markeredgecolor='black', label='Leader (broadcasts "follow me")'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=12, markeredgecolor='black', label='Follower (follows a leader)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                         framealpha=0.9, title='Agent Roles')

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


def create_comparison_plot(baseline_metrics, comm_metrics):
    comparison_dir = BASE_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Communication Comparison: Baseline vs Discrete Communication", fontsize=16, fontweight='bold')

    colors = {"baseline": "steelblue", "discrete": "coral"}

    # Plot 1: Final Reward
    ax = axes[0, 0]
    names = ["Baseline", "Discrete Comm"]
    rewards = [np.mean(baseline_metrics["episode_rewards"][-20:]),
               np.mean(comm_metrics["episode_rewards"][-20:])]
    bars = ax.bar(names, rewards, color=[colors["baseline"], colors["discrete"]], edgecolor='black')
    ax.set_ylabel("Mean Episode Reward (last 20)")
    ax.set_title("Final Performance")
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{r:.0f}', ha='center', fontweight='bold')

    # Plot 2: Gini
    ax = axes[0, 1]
    gini = [np.mean(baseline_metrics["gini_coefficients"][-20:]),
            np.mean(comm_metrics["gini_coefficients"][-20:])]
    bars = ax.bar(names, gini, color=[colors["baseline"], colors["discrete"]], edgecolor='black')
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)
    for bar, g in zip(bars, gini):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{g:.3f}', ha='center', fontweight='bold')

    # Plot 3: Learning Curves
    ax = axes[1, 0]
    for m, c, label in [(baseline_metrics, colors["baseline"], "Baseline"),
                        (comm_metrics, colors["discrete"], "Discrete Comm")]:
        r = m["episode_rewards"]
        window = max(1, len(r) // 20)
        smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=label, color=c, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f'''
    COMMUNICATION COMPARISON
    ========================

    Baseline (No Communication):
      Reward: {rewards[0]:.0f}  |  Gini: {gini[0]:.3f}

    Discrete Communication:
      Reward: {rewards[1]:.0f}  |  Gini: {gini[1]:.3f}

    Messages:
      - "Follow me": Leaders broadcast availability
      - "Full": Leaders reject extra followers

    Improvement: {((rewards[1] - rewards[0]) / rewards[0] * 100):.1f}%
    '''
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_dir / "communication_comparison.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved: {comparison_dir / 'communication_comparison.png'}")


def train_continuous_comm(config, name="Continuous_Comm"):
    """Train with continuous learned communication embeddings."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Communication: CONTINUOUS (learned embeddings)")
    print(f"Embedding dim: 8")
    print(f"Total frames: {config['total_frames']}")
    print(f"{'='*60}")

    env = MingleEnv(
        n_agents=config["n_agents"],
        n_rooms=config["n_rooms"],
        room_capacity=config["room_capacity"],
        max_steps=300,
        phase_mode="claiming",
        reward_modules=get_reward_modules(),
    )

    obs_dim = 14  # Base observation (no discrete comm features)
    n_agents = config["n_agents"]
    comm_dim = 8

    policy = ContinuousCommPolicy(obs_dim, n_agents=n_agents, comm_dim=comm_dim).to(DEVICE)
    critic = Critic(obs_dim + comm_dim).to(DEVICE)  # Critic also sees aggregated messages
    obs_normalizer = ObservationNormalizer(obs_dim, DEVICE)

    policy_optim = torch.optim.AdamW(policy.parameters(), lr=config["lr"], weight_decay=0.01)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config["lr"], weight_decay=0.01)

    metrics = {
        "name": name,
        "communication": "continuous",
        "comm_dim": comm_dim,
        "episode_rewards": [],
        "gini_coefficients": [],
        "losses": [],
    }

    frames = 0
    start_time = time.time()

    while frames < config["total_frames"]:
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        all_messages = []

        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        episode_reward = 0
        agent_episode_rewards = torch.zeros(n_agents)
        epsilon = max(0.1, 0.5 * (1 - frames / config["total_frames"]))

        for _ in range(config["frames_per_batch"]):
            obs_normalizer.update(obs_raw.unsqueeze(0))
            obs = obs_normalizer.normalize(obs_raw)

            with torch.no_grad():
                loc, scale, messages = policy(obs.unsqueeze(0), return_messages=True)
                scale = scale.clamp(0.2, 1.0)
                dist = torch.distributions.Normal(loc, scale)
                action = dist.sample()

                if torch.rand(1).item() < epsilon:
                    action = torch.rand_like(action) * 0.6 - 0.3

                action = action.clamp(-0.3, 0.3)
                log_prob = dist.log_prob(action).sum(dim=-1)

                # Aggregate messages for critic
                agg_msgs = policy.aggregate_messages(messages.unsqueeze(0)).squeeze(0)
                obs_with_msgs = torch.cat([obs, agg_msgs], dim=-1)
                value = critic(obs_with_msgs.unsqueeze(0))

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
            all_messages.append(messages)

            episode_reward += reward.sum().item()
            agent_episode_rewards += reward.squeeze().cpu()

            if done:
                metrics["episode_rewards"].append(episode_reward)
                metrics["gini_coefficients"].append(compute_gini(agent_episode_rewards.numpy()))
                episode_reward = 0
                agent_episode_rewards = torch.zeros(n_agents)
                td = env.reset()
                obs_raw = td["observation"].to(DEVICE)
            else:
                obs_raw = next_td["observation"].to(DEVICE)

        frames += config["frames_per_batch"] * n_agents

        # PPO update
        obs_tensor = torch.stack(observations)
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.stack(rewards)
        dones_tensor = torch.tensor(dones, device=DEVICE)
        old_log_probs = torch.stack(log_probs)
        values_tensor = torch.stack(values)

        with torch.no_grad():
            obs = obs_normalizer.normalize(obs_raw)
            _, _, next_msgs = policy(obs.unsqueeze(0), return_messages=True)
            agg_msgs = policy.aggregate_messages(next_msgs.unsqueeze(0)).squeeze(0)
            obs_with_msgs = torch.cat([obs, agg_msgs], dim=-1)
            next_value = critic(obs_with_msgs.unsqueeze(0)).squeeze()

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

        T = len(rewards_tensor)
        obs_flat = obs_tensor.view(T * n_agents, -1)
        actions_flat = actions_tensor.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        total_loss = 0
        for _ in range(config["num_epochs"]):
            loc, scale, msgs = policy(obs_flat, return_messages=True)
            scale = scale.clamp(0.1, 1.0)
            dist = torch.distributions.Normal(loc, scale)
            new_log_probs = dist.log_prob(actions_flat).sum(dim=-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - config["clip_epsilon"], 1 + config["clip_epsilon"]) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Reshape for critic
            msgs_reshaped = msgs.view(T, n_agents, -1)
            agg_msgs = policy.aggregate_messages(msgs_reshaped).view(T * n_agents, -1)
            obs_with_msgs = torch.cat([obs_flat, agg_msgs], dim=-1)
            new_values = critic(obs_with_msgs.view(T, n_agents, -1)).view(-1)
            value_loss = nn.functional.mse_loss(new_values, returns_flat)

            loss = policy_loss + 0.5 * value_loss - config["entropy_coef"] * entropy

            policy_optim.zero_grad()
            critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            policy_optim.step()
            critic_optim.step()
            total_loss += loss.item()

        metrics["losses"].append(total_loss / config["num_epochs"])

        if len(metrics["episode_rewards"]) > 0 and frames % 20000 < config["frames_per_batch"] * n_agents:
            recent_rewards = metrics["episode_rewards"][-10:]
            recent_gini = metrics["gini_coefficients"][-10:]
            print(f"Frames: {frames:8d}/{config['total_frames']} | "
                  f"Reward: {np.mean(recent_rewards):8.2f} | "
                  f"Gini: {np.mean(recent_gini):.3f} | "
                  f"Eps: {epsilon:.2f}")

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")

    # Save results
    output_dir = BASE_DIR / "continuous"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
    }, output_dir / "model.pt")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")

    # Generate GIF
    generate_continuous_gif(env, policy, obs_normalizer, output_dir / "trained_policy.gif", name, n_agents)

    return metrics


def generate_continuous_gif(env, policy, obs_normalizer, output_path, title, n_agents):
    """Generate GIF for continuous communication."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        for step in range(200):
            obs_normalizer.update(obs_raw.unsqueeze(0))
            obs = obs_normalizer.normalize(obs_raw)

            with torch.no_grad():
                loc, scale, messages = policy(obs.unsqueeze(0), return_messages=True)
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
                ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, marker='o',
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(f"A{i}", (pos[0], pos[1] + 0.7), ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Legend for continuous communication
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=12, markeredgecolor='black', label='Agent (sends 8D embedding)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                     framealpha=0.9, title='Continuous Comm')

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


def create_full_comparison_plot(baseline_metrics, discrete_metrics, continuous_metrics):
    """Create comparison plot for all three communication methods."""
    comparison_dir = BASE_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Communication Comparison: Baseline vs Discrete vs Continuous", fontsize=16, fontweight='bold')

    colors = {"baseline": "steelblue", "discrete": "coral", "continuous": "forestgreen"}
    names = ["Baseline", "Discrete", "Continuous"]
    all_metrics = [baseline_metrics, discrete_metrics, continuous_metrics]

    # Plot 1: Final Reward
    ax = axes[0, 0]
    rewards = [np.mean(m["episode_rewards"][-20:]) for m in all_metrics]
    bars = ax.bar(names, rewards, color=[colors["baseline"], colors["discrete"], colors["continuous"]], edgecolor='black')
    ax.set_ylabel("Mean Episode Reward (last 20)")
    ax.set_title("Final Performance")
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{r:.0f}', ha='center', fontweight='bold')

    # Plot 2: Gini
    ax = axes[0, 1]
    gini = [np.mean(m["gini_coefficients"][-20:]) for m in all_metrics]
    bars = ax.bar(names, gini, color=[colors["baseline"], colors["discrete"], colors["continuous"]], edgecolor='black')
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)
    for bar, g in zip(bars, gini):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{g:.3f}', ha='center', fontweight='bold')

    # Plot 3: Learning Curves
    ax = axes[1, 0]
    for m, c, label in [(baseline_metrics, colors["baseline"], "Baseline"),
                        (discrete_metrics, colors["discrete"], "Discrete"),
                        (continuous_metrics, colors["continuous"], "Continuous")]:
        r = m["episode_rewards"]
        window = max(1, len(r) // 20)
        smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=label, color=c, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f'''
    COMMUNICATION COMPARISON
    ========================

    Baseline (No Communication):
      Reward: {rewards[0]:.0f}  |  Gini: {gini[0]:.3f}

    Discrete Communication (10 pts):
      Reward: {rewards[1]:.0f}  |  Gini: {gini[1]:.3f}
      Messages: "follow_me" / "full"

    Continuous Communication (5 pts):
      Reward: {rewards[2]:.0f}  |  Gini: {gini[2]:.3f}
      Embedding: 8-dimensional learned vector

    Best Reward: {names[np.argmax(rewards)]}
    Best Fairness: {names[np.argmin(gini)]}
    '''
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_dir / "communication_comparison.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved: {comparison_dir / 'communication_comparison.png'}")

    # Save results
    results = {
        "Baseline": {"reward": rewards[0], "gini": gini[0]},
        "Discrete": {"reward": rewards[1], "gini": gini[1]},
        "Continuous": {"reward": rewards[2], "gini": gini[2]},
    }
    with open(comparison_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    print("="*70)
    print("COMMUNICATION COMPARISON: Baseline vs Discrete vs Continuous")
    print("="*70)
    print(f"Device: {DEVICE}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "n_agents": 6,
        "n_rooms": 3,
        "room_capacity": 2,
        "total_frames": 300000,
        "frames_per_batch": 2048,
        "num_epochs": 4,
        "lr": 3e-4,
        "gamma": 0.99,
        "lmbda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.1,
    }

    # Train baseline (no communication)
    print("\n" + "="*70)
    print("1/3: TRAINING BASELINE (No Communication)")
    print("="*70)
    baseline_metrics = train_communication(False, config, "Baseline")

    # Train with discrete communication
    print("\n" + "="*70)
    print("2/3: TRAINING DISCRETE COMMUNICATION")
    print("="*70)
    discrete_metrics = train_communication(True, config, "Discrete_Comm")

    # Train with continuous communication
    print("\n" + "="*70)
    print("3/3: TRAINING CONTINUOUS COMMUNICATION")
    print("="*70)
    continuous_metrics = train_continuous_comm(config, "Continuous_Comm")

    # Create comparison plot
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    create_full_comparison_plot(baseline_metrics, discrete_metrics, continuous_metrics)

    print("\n" + "="*70)
    print("COMMUNICATION COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {BASE_DIR}")
    print("  - baseline/")
    print("  - discrete_comm/")
    print("  - continuous/")
    print("  - comparison/communication_comparison.png")


if __name__ == "__main__":
    main()
