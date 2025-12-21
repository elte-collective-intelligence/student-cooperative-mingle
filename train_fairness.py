"""
Fairness Comparison: Baseline vs Gini vs Participation

Three fairness approaches:
1. Baseline: No fairness mechanism, pure reward maximization
2. Gini: Penalizes unequal reward distribution (lower Gini = fairer)
3. Participation: Penalizes unequal room visitation across agents

Usage:
    python train_fairness.py
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
BASE_DIR = Path("contribution_tests_and_comparisions/fairness")


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


class FairnessPolicy(nn.Module):
    """Policy for fairness experiments."""
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


def train_fairness_method(method_name, config, gini_weight=0.0, participation_weight=0.0):
    """Train with specific fairness mechanism."""
    print(f"\n{'='*60}")
    print(f"Training: {method_name}")
    print(f"Gini weight: {gini_weight}, Participation weight: {participation_weight}")
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

    obs_dim = 14
    n_agents = config["n_agents"]

    policy = FairnessPolicy(obs_dim, n_agents=n_agents).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    obs_normalizer = ObservationNormalizer(obs_dim, DEVICE)

    policy_optim = torch.optim.AdamW(policy.parameters(), lr=config["lr"], weight_decay=0.01)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config["lr"], weight_decay=0.01)

    # Track room visits per agent for participation fairness
    agent_room_visits = torch.zeros(n_agents, device=DEVICE)

    metrics = {
        "name": method_name,
        "gini_weight": gini_weight,
        "participation_weight": participation_weight,
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
        agent_room_visits.zero_()

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

            # Track room visits for participation
            if hasattr(env, 'agent_current_room'):
                in_room = env.agent_current_room >= 0
                agent_room_visits += in_room.float()

            # Apply fairness modifications to reward
            if gini_weight > 0:
                # Gini penalty: penalize unequal reward distribution
                current_rewards = agent_episode_rewards + reward.squeeze().cpu()
                if current_rewards.sum() > 0:
                    gini = compute_gini(current_rewards.numpy())
                    gini_penalty = gini_weight * gini
                    reward = reward - gini_penalty

            if participation_weight > 0:
                # Participation penalty: penalize unequal room visits
                if agent_room_visits.sum() > 0:
                    visits_normalized = agent_room_visits / (agent_room_visits.sum() + 1e-8)
                    expected = 1.0 / n_agents
                    variance = ((visits_normalized - expected) ** 2).mean()
                    participation_penalty = participation_weight * variance.item()
                    reward = reward - participation_penalty

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
                agent_room_visits.zero_()
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

        if len(metrics["episode_rewards"]) > 0 and frames % 60000 < config["frames_per_batch"] * n_agents:
            recent_rewards = metrics["episode_rewards"][-10:]
            recent_gini = metrics["gini_coefficients"][-10:]
            print(f"  Frames: {frames:6d}/{config['total_frames']} | "
                  f"Loss: {total_loss/config['num_epochs']:10.4f} | "
                  f"Reward: {np.mean(recent_rewards):8.2f} | "
                  f"Gini: {np.mean(recent_gini):.3f}")

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")

    # Save results
    folder_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = BASE_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
    }, output_dir / "model.pt")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")

    # Generate GIF with legend
    generate_gif_with_legend(env, policy, obs_normalizer, output_dir / "trained_policy.gif",
                             method_name, n_agents, gini_weight, participation_weight)

    return metrics, policy, obs_normalizer


def generate_gif_with_legend(env, policy, obs_normalizer, output_path, title, n_agents, gini_weight, participation_weight):
    """Generate GIF with fairness method legend."""
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')

        gif_frames = []
        td = env.reset()
        obs_raw = td["observation"].to(DEVICE)

        # Determine legend text based on fairness method
        if gini_weight > 0:
            legend_text = f"Gini Fairness\nPenalizes unequal reward\ndistribution (weight={gini_weight})"
            box_color = 'lightgreen'
        elif participation_weight > 0:
            legend_text = f"Participation Fairness\nPenalizes unequal room\nvisitation (weight={participation_weight})"
            box_color = 'lightblue'
        else:
            legend_text = "Baseline\nNo fairness mechanism\nPure reward maximization"
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
                ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, marker='o',
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(f"A{i}", (pos[0], pos[1] + 0.7), ha='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Add legend box with method description
            ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9, edgecolor='gray'))

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


def create_comparison_plot(all_metrics):
    """Create comparison plot for all fairness methods."""
    comparison_dir = BASE_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fairness Comparison: Baseline vs Gini vs Participation", fontsize=16, fontweight='bold')

    colors = {"baseline": "steelblue", "gini": "forestgreen", "participation": "coral"}
    names = list(all_metrics.keys())

    # Plot 1: Final Reward
    ax = axes[0, 0]
    rewards = [np.mean(m["episode_rewards"][-20:]) for m in all_metrics.values()]
    bars = ax.bar(names, rewards, color=[colors.get(n.lower(), "gray") for n in names], edgecolor='black')
    ax.set_ylabel("Mean Episode Reward (last 20)")
    ax.set_title("Final Performance")
    ax.grid(axis='y', alpha=0.3)
    for bar, r in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{r:.0f}', ha='center', fontweight='bold')

    # Plot 2: Gini Coefficient
    ax = axes[0, 1]
    ginis = [np.mean(m["gini_coefficients"][-20:]) for m in all_metrics.values()]
    bars = ax.bar(names, ginis, color=[colors.get(n.lower(), "gray") for n in names], edgecolor='black')
    ax.set_ylabel("Gini Coefficient (lower = fairer)")
    ax.set_title("Fairness Comparison")
    ax.grid(axis='y', alpha=0.3)
    for bar, g in zip(bars, ginis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{g:.3f}', ha='center', fontweight='bold')

    # Plot 3: Learning Curves
    ax = axes[1, 0]
    for name, m in all_metrics.items():
        r = m["episode_rewards"]
        window = max(1, len(r) // 20)
        smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, color=colors.get(name.lower(), "gray"), linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')

    baseline_reward = rewards[0]
    summary_lines = ["FAIRNESS COMPARISON SUMMARY", "=" * 35, ""]

    for i, (name, m) in enumerate(all_metrics.items()):
        improvement = ((rewards[i] - baseline_reward) / baseline_reward * 100) if baseline_reward > 0 else 0
        gini_improvement = ((ginis[0] - ginis[i]) / ginis[0] * 100) if ginis[0] > 0 else 0
        summary_lines.append(f"{name}:")
        summary_lines.append(f"  Reward: {rewards[i]:.0f} ({improvement:+.1f}%)")
        summary_lines.append(f"  Gini: {ginis[i]:.3f} ({gini_improvement:+.1f}% fairer)")
        summary_lines.append("")

    summary_lines.append("Lower Gini = More equal distribution")

    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_dir / "fairness_comparison.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved: {comparison_dir / 'fairness_comparison.png'}")

    # Save comparison results
    results = {name: {"final_reward": rewards[i], "final_gini": ginis[i]}
               for i, name in enumerate(names)}
    with open(comparison_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {comparison_dir / 'comparison_results.json'}")


def main():
    print("=" * 70)
    print("FAIRNESS COMPARISON: Baseline vs Gini vs Participation")
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

    all_metrics = {}

    # 1. Train Baseline (no fairness)
    print("\n" + "=" * 70)
    print("1/3: TRAINING BASELINE (No Fairness)")
    print("=" * 70)
    metrics, _, _ = train_fairness_method("Baseline", config, gini_weight=0.0, participation_weight=0.0)
    all_metrics["Baseline"] = metrics

    # 2. Train Gini Fairness
    print("\n" + "=" * 70)
    print("2/3: TRAINING GINI FAIRNESS")
    print("=" * 70)
    metrics, _, _ = train_fairness_method("Gini", config, gini_weight=0.5, participation_weight=0.0)
    all_metrics["Gini"] = metrics

    # 3. Train Participation Fairness
    print("\n" + "=" * 70)
    print("3/3: TRAINING PARTICIPATION FAIRNESS")
    print("=" * 70)
    metrics, _, _ = train_fairness_method("Participation", config, gini_weight=0.0, participation_weight=0.5)
    all_metrics["Participation"] = metrics

    # Create comparison plot
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    create_comparison_plot(all_metrics)

    print("\n" + "=" * 70)
    print("FAIRNESS COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {BASE_DIR}")
    print("  - baseline/model.pt, metrics.json, trained_policy.gif")
    print("  - gini/model.pt, metrics.json, trained_policy.gif")
    print("  - participation/model.pt, metrics.json, trained_policy.gif")
    print("  - comparison/fairness_comparison.png, comparison_results.json")


if __name__ == "__main__":
    main()
