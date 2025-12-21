"""
Full Stack Comparison: Baseline vs Full Stack (All Contributions)

Full Stack combines ALL contributions:
1. MAPPO (Multi-Agent PPO with centralized critic)
2. Discrete Communication (leader-follower with "follow me"/"full" messages)
3. Gini Fairness (penalizes unequal reward distribution)
4. Curriculum Learning (progressive difficulty)

This demonstrates the synergistic effect of combining all improvements.

Usage:
    python train_fullstack.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
BASE_DIR = Path("contribution_tests_and_comparisions/fullstack")


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


class FullStackPolicy(nn.Module):
    """Policy with communication support for full stack."""
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

        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                was_flat = True
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
            else:
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


class MAPPOCritic(nn.Module):
    """MAPPO Centralized critic - sees ALL agents' observations."""
    def __init__(self, obs_dim, n_agents, hidden_dim=256):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        obs_concat = obs.view(batch_size, -1)
        values = self.net(obs_concat)
        return values.squeeze(0)


def get_reward_modules(overfill_penalty=5.0):
    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=overfill_penalty, phase_mode="claiming"),
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


def train_method(method_name, config, use_mappo=False, use_communication=False,
                 use_gini=False, use_curriculum=False):
    """Train with specified contributions."""

    print(f"\n{'='*60}")
    print(f"Training: {method_name}")
    print(f"  MAPPO: {'ON' if use_mappo else 'OFF'}")
    print(f"  Communication: {'ON' if use_communication else 'OFF'}")
    print(f"  Gini Fairness: {'ON' if use_gini else 'OFF'}")
    print(f"  Curriculum: {'ON' if use_curriculum else 'OFF'}")
    print(f"Total frames: {config['total_frames']}")
    print(f"{'='*60}")

    n_agents = config["n_agents"]
    obs_dim = 18 if use_communication else 14

    policy = FullStackPolicy(obs_dim, n_agents=n_agents).to(DEVICE)

    if use_mappo:
        critic = MAPPOCritic(obs_dim, n_agents).to(DEVICE)
    else:
        # Simple shared critic
        critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(DEVICE)

    obs_normalizer = ObservationNormalizer(obs_dim, DEVICE)

    policy_optim = torch.optim.AdamW(policy.parameters(), lr=config["lr"], weight_decay=0.01)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config["lr"], weight_decay=0.01)

    # Curriculum stages
    if use_curriculum:
        stages = [
            {"name": "Easy", "room_capacity": 3, "overfill_penalty": 2.0, "frames": config["total_frames"] // 3},
            {"name": "Medium", "room_capacity": 2, "overfill_penalty": 5.0, "frames": config["total_frames"] // 3},
            {"name": "Hard", "room_capacity": 2, "overfill_penalty": 8.0, "frames": config["total_frames"] // 3},
        ]
    else:
        stages = [
            {"name": "Standard", "room_capacity": 2, "overfill_penalty": 5.0, "frames": config["total_frames"]},
        ]

    metrics = {
        "name": method_name,
        "mappo": use_mappo,
        "communication": use_communication,
        "gini_fairness": use_gini,
        "curriculum": use_curriculum,
        "episode_rewards": [],
        "gini_coefficients": [],
        "losses": [],
    }

    total_frames = 0
    start_time = time.time()

    for stage_idx, stage in enumerate(stages):
        if use_curriculum:
            print(f"\n--- Stage {stage_idx + 1}/{len(stages)}: {stage['name']} ---")

        env = MingleEnv(
            n_agents=config["n_agents"],
            n_rooms=config["n_rooms"],
            room_capacity=stage["room_capacity"],
            max_steps=300,
            phase_mode="claiming",
            reward_modules=get_reward_modules(stage["overfill_penalty"]),
        )
        env.enable_communication = use_communication

        stage_frames = 0

        while stage_frames < stage["frames"]:
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
            epsilon = max(0.1, 0.5 * (1 - total_frames / config["total_frames"]))

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

                    if use_mappo:
                        value = critic(obs.unsqueeze(0))
                    else:
                        obs_flat = obs.view(-1, obs_dim)
                        value = critic(obs_flat).squeeze(-1)

                step_td = TensorDict({"action": action.cpu()}, batch_size=[])
                next_td = env._step(step_td)

                reward = next_td["reward"].to(DEVICE)
                done = next_td["done"].any() or next_td["terminated"].any()

                # Apply Gini fairness penalty
                if use_gini:
                    current_rewards = agent_episode_rewards + reward.squeeze().cpu()
                    if current_rewards.sum() > 0:
                        gini = compute_gini(current_rewards.numpy())
                        gini_penalty = 0.5 * gini
                        reward = reward - gini_penalty

                observations.append(obs)
                actions.append(action)
                rewards.append(reward.squeeze())
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value.squeeze() if use_mappo else value)

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

            stage_frames += config["frames_per_batch"] * n_agents
            total_frames += config["frames_per_batch"] * n_agents

            # PPO update
            obs_tensor = torch.stack(observations)
            actions_tensor = torch.stack(actions)
            rewards_tensor = torch.stack(rewards)
            dones_tensor = torch.tensor(dones, device=DEVICE)
            old_log_probs = torch.stack(log_probs)
            values_tensor = torch.stack(values)

            with torch.no_grad():
                obs = obs_normalizer.normalize(obs_raw)
                if use_mappo:
                    next_value = critic(obs.unsqueeze(0)).squeeze()
                else:
                    obs_flat = obs.view(-1, obs_dim)
                    next_value = critic(obs_flat).squeeze(-1)

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

                if use_mappo:
                    new_values = critic(obs_tensor).view(-1)
                else:
                    new_values = critic(obs_flat).squeeze(-1)
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

            if len(metrics["episode_rewards"]) > 0 and total_frames % 60000 < config["frames_per_batch"] * n_agents:
                recent_rewards = metrics["episode_rewards"][-10:]
                recent_gini = metrics["gini_coefficients"][-10:]
                print(f"  Frames: {total_frames:6d}/{config['total_frames']} | "
                      f"Reward: {np.mean(recent_rewards):8.2f} | "
                      f"Gini: {np.mean(recent_gini):.3f}")

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")

    # Save results
    folder_name = method_name.lower().replace(" ", "_")
    output_dir = BASE_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict() if use_mappo else None,
    }, output_dir / "model.pt")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")

    # Create environment for GIF
    env = MingleEnv(
        n_agents=config["n_agents"],
        n_rooms=config["n_rooms"],
        room_capacity=2,
        max_steps=300,
        phase_mode="claiming",
        reward_modules=get_reward_modules(5.0),
    )
    env.enable_communication = use_communication

    generate_gif(env, policy, obs_normalizer, output_dir / "trained_policy.gif",
                 method_name, n_agents, use_communication,
                 use_mappo, use_gini, use_curriculum)

    return metrics


def generate_gif(env, policy, obs_normalizer, output_path, title, n_agents,
                 use_communication, use_mappo, use_gini, use_curriculum):
    """Generate GIF with full stack legend."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        # Build legend text
        features = []
        if use_mappo:
            features.append("MAPPO (centralized critic)")
        if use_communication:
            features.append("Communication (leader/follower)")
        if use_gini:
            features.append("Gini fairness penalty")
        if use_curriculum:
            features.append("Curriculum learning")

        if features:
            legend_text = "Full Stack:\n" + "\n".join([f"+ {f}" for f in features])
            box_color = 'lightgreen'
        else:
            legend_text = "Baseline\nNo contributions enabled"
            box_color = 'lightyellow'

        for step in range(200):
            obs_normalizer.update(obs_raw.unsqueeze(0))
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
                if use_communication and env.is_leader[i]:
                    marker = 's'
                    label = f"L{i}"
                else:
                    marker = 'o'
                    label = f"A{i}" if not use_communication else f"F{i}"

                ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, marker=marker,
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(label, (pos[0], pos[1] + 0.7), ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Add legend
            ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9, edgecolor='gray'))

            if use_communication:
                legend_elements = [
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                           markersize=10, markeredgecolor='black', label='Leader'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=10, markeredgecolor='black', label='Follower')
                ]
                ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

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


def create_comparison_plot(baseline_metrics, fullstack_metrics):
    """Create comparison plot."""
    comparison_dir = BASE_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Full Stack Comparison: Baseline vs All Contributions Combined", fontsize=16, fontweight='bold')

    colors = {"baseline": "steelblue", "fullstack": "forestgreen"}

    # Plot 1: Final Reward
    ax = axes[0, 0]
    names = ["Baseline", "Full Stack"]
    rewards = [np.mean(baseline_metrics["episode_rewards"][-20:]),
               np.mean(fullstack_metrics["episode_rewards"][-20:])]
    bars = ax.bar(names, rewards, color=[colors["baseline"], colors["fullstack"]], edgecolor='black')
    ax.set_ylabel("Mean Episode Reward (last 20)")
    ax.set_title("Final Performance")
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{r:.0f}', ha='center', fontweight='bold')

    # Plot 2: Gini
    ax = axes[0, 1]
    gini = [np.mean(baseline_metrics["gini_coefficients"][-20:]),
            np.mean(fullstack_metrics["gini_coefficients"][-20:])]
    bars = ax.bar(names, gini, color=[colors["baseline"], colors["fullstack"]], edgecolor='black')
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)
    for bar, g in zip(bars, gini):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{g:.3f}', ha='center', fontweight='bold')

    # Plot 3: Learning Curves
    ax = axes[1, 0]
    for m, c, label in [(baseline_metrics, colors["baseline"], "Baseline"),
                        (fullstack_metrics, colors["fullstack"], "Full Stack")]:
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
    improvement = ((rewards[1] - rewards[0]) / rewards[0] * 100) if rewards[0] > 0 else 0
    gini_improvement = ((gini[0] - gini[1]) / gini[0] * 100) if gini[0] > 0 else 0

    summary = f'''
    FULL STACK COMPARISON
    =====================

    Baseline:
      Reward: {rewards[0]:.0f}  |  Gini: {gini[0]:.3f}

    Full Stack (ALL contributions):
      Reward: {rewards[1]:.0f}  |  Gini: {gini[1]:.3f}

    Contributions Combined:
      + MAPPO (centralized critic)
      + Discrete Communication
      + Gini Fairness
      + Curriculum Learning

    IMPROVEMENT:
      Reward: {improvement:+.1f}%
      Fairness: {gini_improvement:+.1f}% better
    '''
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_dir / "fullstack_comparison.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved: {comparison_dir / 'fullstack_comparison.png'}")

    results = {
        "Baseline": {"final_reward": rewards[0], "final_gini": gini[0]},
        "Full_Stack": {"final_reward": rewards[1], "final_gini": gini[1]},
        "improvement_reward_pct": improvement,
        "improvement_gini_pct": gini_improvement,
    }
    with open(comparison_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    print("=" * 70)
    print("FULL STACK COMPARISON: Baseline vs All Contributions")
    print("=" * 70)
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

    # Train baseline (no contributions)
    print("\n" + "=" * 70)
    print("1/2: TRAINING BASELINE (No Contributions)")
    print("=" * 70)
    baseline_metrics = train_method("Baseline", config,
                                    use_mappo=False,
                                    use_communication=False,
                                    use_gini=False,
                                    use_curriculum=False)

    # Train full stack (all contributions)
    print("\n" + "=" * 70)
    print("2/2: TRAINING FULL STACK (All Contributions)")
    print("=" * 70)
    fullstack_metrics = train_method("Full Stack", config,
                                     use_mappo=True,
                                     use_communication=True,
                                     use_gini=True,
                                     use_curriculum=True)

    # Create comparison
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    create_comparison_plot(baseline_metrics, fullstack_metrics)

    print("\n" + "=" * 70)
    print("FULL STACK COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {BASE_DIR}")
    print("  - baseline/model.pt, metrics.json, trained_policy.gif")
    print("  - full_stack/model.pt, metrics.json, trained_policy.gif")
    print("  - comparison/fullstack_comparison.png, comparison_results.json")


if __name__ == "__main__":
    main()
