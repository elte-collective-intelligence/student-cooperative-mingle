"""
Train models with CURRICULUM LEARNING and generate GIFs with trained policies.

Curriculum:
- Stage 1 (Easy): 2 agents, 1 room, capacity 2
- Stage 2 (Medium): 4 agents, 2 rooms, capacity 2
- Stage 3 (Hard): 6 agents, 3 rooms, capacity 2

Final visualization uses the hard configuration (6 agents, 3 rooms).
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from pathlib import Path
from datetime import datetime
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.mingle_env import MingleEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Curriculum stages: Easy -> Medium -> Hard
CURRICULUM = [
    {"name": "easy", "n_agents": 2, "n_rooms": 1, "room_capacity": 2, "frames": 20000},
    {"name": "medium", "n_agents": 4, "n_rooms": 2, "room_capacity": 2, "frames": 30000},
    {"name": "hard", "n_agents": 6, "n_rooms": 3, "room_capacity": 2, "frames": 50000},
]

# Final config for visualization
FINAL_CONFIG = {
    "n_agents": 6,
    "n_rooms": 3,
    "room_capacity": 2,
}

# Training hyperparameters
TRAIN_CONFIG = {
    "frames_per_batch": 500,
    "num_epochs": 4,
    "lr": 3e-4,
    "gamma": 0.99,
    "lmbda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "hidden_dim": 128,
}


class PolicyNetwork(nn.Module):
    """Policy network - handles variable agent counts via padding."""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.min_std = 0.1

    def forward(self, obs):
        # Ensure obs has correct dimension
        if obs.dim() == 2:  # (n_agents, obs_dim)
            obs = obs.unsqueeze(0)  # (1, n_agents, obs_dim)

        batch, n_agents, obs_dim = obs.shape

        # Pad or truncate obs_dim if needed
        if obs_dim != self.obs_dim:
            if obs_dim < self.obs_dim:
                pad = torch.zeros(batch, n_agents, self.obs_dim - obs_dim, device=obs.device)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                obs = obs[..., :self.obs_dim]

        obs_flat = obs.view(-1, self.obs_dim)
        h = self.net(obs_flat)
        mean = self.mean(h)
        std = torch.exp(self.log_std(h)).clamp(min=self.min_std)
        return mean.view(batch, n_agents, -1), std.view(batch, n_agents, -1)


class CriticNetwork(nn.Module):
    """Value network."""
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
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

        batch, n_agents, obs_dim = obs.shape

        if obs_dim != self.obs_dim:
            if obs_dim < self.obs_dim:
                pad = torch.zeros(batch, n_agents, self.obs_dim - obs_dim, device=obs.device)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                obs = obs[..., :self.obs_dim]

        obs_flat = obs.view(-1, self.obs_dim)
        v = self.net(obs_flat)
        return v.view(batch, n_agents, 1)


class CurriculumPPOTrainer:
    """PPO trainer with curriculum learning support."""

    def __init__(self, config, device, name="ppo"):
        self.config = config
        self.device = device
        self.name = name

        # Use max obs_dim from hardest stage (6 agents, 3 rooms)
        # obs_dim = 2 (pos) + 2 (vel) + 2*n_rooms (room_pos) + n_rooms (room_occ) + 1 (phase) + ...
        # Estimate: ~20 for 3 rooms
        self.obs_dim = 20
        action_dim = 2

        self.policy = PolicyNetwork(self.obs_dim, action_dim, config["hidden_dim"]).to(device)
        self.critic = CriticNetwork(self.obs_dim, config["hidden_dim"]).to(device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=config["lr"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config["lr"])

        self.metrics = {"rewards": [], "losses": [], "stages": []}
        self.current_env = None

    def set_env(self, env):
        """Set current environment for training."""
        self.current_env = env

    def collect_rollout(self, num_steps):
        """Collect experience."""
        env = self.current_env
        observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        td = env.reset()
        obs = td["observation"].to(self.device)

        for _ in range(num_steps):
            with torch.no_grad():
                mean, std = self.policy(obs)
                mean, std = mean.squeeze(0), std.squeeze(0)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().clamp(-env.max_speed, env.max_speed)
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(obs).squeeze(0)

            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            reward = next_td["reward"].to(self.device)
            done = next_td["done"].any() or next_td["terminated"].any()

            observations.append(obs)
            actions.append(action)
            rewards.append(reward.squeeze())
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.squeeze())

            if done:
                td = env.reset()
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

    def compute_gae(self, rewards, values, dones, gamma=0.99, lmbda=0.95):
        """Compute GAE."""
        T, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(n_agents, device=rewards.device)

        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else torch.zeros(n_agents, device=rewards.device)
            mask = 1 - dones[t].float()
            delta = rewards[t] + gamma * next_val * mask - values[t]
            advantages[t] = delta + gamma * lmbda * mask * last_adv
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

        advantages, returns = self.compute_gae(rewards, values, dones,
                                                self.config["gamma"], self.config["lmbda"])

        T, n_agents = rewards.shape
        obs_flat = obs.view(T * n_agents, -1)

        # Pad obs if needed
        if obs_flat.shape[-1] < self.obs_dim:
            pad = torch.zeros(obs_flat.shape[0], self.obs_dim - obs_flat.shape[-1], device=self.device)
            obs_flat = torch.cat([obs_flat, pad], dim=-1)

        actions_flat = actions.view(T * n_agents, -1)
        old_log_probs_flat = old_log_probs.view(T * n_agents)
        advantages_flat = advantages.view(T * n_agents)
        returns_flat = returns.view(T * n_agents)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        total_loss = 0
        for _ in range(self.config["num_epochs"]):
            mean, std = self.policy(obs_flat.unsqueeze(1))
            mean, std = mean.squeeze(1), std.squeeze(1)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_flat).sum(dim=-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - self.config["clip_epsilon"],
                               1 + self.config["clip_epsilon"]) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(obs_flat.unsqueeze(1)).squeeze()
            value_loss = nn.functional.mse_loss(new_values, returns_flat)

            loss = policy_loss + 0.5 * value_loss - self.config["entropy_coef"] * entropy

            self.policy_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.policy_optim.step()
            self.critic_optim.step()

            total_loss += loss.item()

        self.metrics["losses"].append(total_loss / self.config["num_epochs"])
        self.metrics["rewards"].append(rewards.mean().item())

        return total_loss / self.config["num_epochs"]

    def train_stage(self, stage):
        """Train on a curriculum stage."""
        print(f"\n  Stage: {stage['name'].upper()} ({stage['n_agents']} agents, {stage['n_rooms']} rooms)")

        env = MingleEnv(
            n_agents=stage["n_agents"],
            n_rooms=stage["n_rooms"],
            room_capacity=stage["room_capacity"],
            max_steps=100
        )
        self.set_env(env)

        frames = 0
        update_count = 0
        total_frames = stage["frames"]

        while frames < total_frames:
            rollout = self.collect_rollout(self.config["frames_per_batch"])
            frames += self.config["frames_per_batch"] * env.n_agents
            loss = self.update(rollout)
            update_count += 1

            if update_count % 5 == 0:
                print(f"    Frames: {frames}/{total_frames} | Loss: {loss:.4f}")

        self.metrics["stages"].append({
            "name": stage["name"],
            "frames": frames,
            "final_loss": self.metrics["losses"][-1] if self.metrics["losses"] else 0
        })

    def train_curriculum(self, curriculum):
        """Train through all curriculum stages."""
        print(f"\n  Training {self.name} with curriculum learning...")
        for stage in curriculum:
            self.train_stage(stage)
        print(f"  Curriculum complete!")

    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_dim": self.obs_dim,
            "metrics": self.metrics,
        }, path)
        print(f"  Model saved: {path}")

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        print(f"  Model loaded: {path}")


def make_gif_with_policy(env, policy, steps=200, gif_path="output.gif", fps=10, title="", device=DEVICE):
    """Generate GIF using trained policy."""
    frames = []
    td = env.reset()
    obs = td["observation"].to(device)

    policy.eval()

    for step in range(steps):
        with torch.no_grad():
            mean, std = policy(obs)
            mean, std = mean.squeeze(0), std.squeeze(0)
            action = mean.clamp(-env.max_speed, env.max_speed)

        step_td = TensorDict({"action": action.cpu()}, batch_size=[])
        td = env._step(step_td)

        reward = td["reward"].sum().item()

        # Create frame
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-env.arena_radius - 2, env.arena_radius + 2)
        ax.set_ylim(-env.arena_radius - 2, env.arena_radius + 2)
        ax.set_aspect('equal')
        ax.set_title(f"{title} | Step {step} | Phase: {env.phase} | Reward: {reward:.2f}", fontsize=12)

        # Arena
        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--", linewidth=2))

        # Center
        center_color = "lightblue" if env.phase == "spinning" else "lightsalmon"
        edge_color = "blue" if env.phase == "spinning" else "red"
        ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color=center_color, alpha=0.3))
        ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color=edge_color, linewidth=2))

        # Rooms
        if env.room_positions is not None:
            room_positions = env.room_positions.cpu().numpy()
            room_occupancy = env.room_occupancy.cpu().numpy() if hasattr(env, 'room_occupancy') else np.zeros(len(room_positions))

            for i, (pos, occ) in enumerate(zip(room_positions, room_occupancy)):
                fill_ratio = min(occ / env.room_capacity, 1.0) if env.room_capacity > 0 else 0
                color = "green" if fill_ratio >= 1.0 else "yellow"
                alpha = 0.5 if fill_ratio >= 1.0 else 0.3

                ax.add_artist(plt.Circle(pos, env.room_radius, fill=True, color=color, alpha=alpha))
                ax.add_artist(plt.Circle(pos, env.room_radius, fill=False, color="green", linewidth=2))
                ax.text(pos[0], pos[1], f"R{i}\n{int(occ)}/{env.room_capacity}",
                       ha='center', va='center', fontsize=10, fontweight='bold')

        # Agents
        positions = env.agent_positions.cpu().numpy()
        in_center = np.linalg.norm(positions, axis=1) <= env.center_radius
        in_room = np.zeros(len(positions), dtype=bool)

        if env.room_positions is not None:
            room_pos = env.room_positions.cpu().numpy()
            for i, pos in enumerate(positions):
                dists = np.linalg.norm(room_pos - pos, axis=1)
                if dists.min() < env.room_radius:
                    in_room[i] = True

        # Different colors for agents
        agent_colors = plt.cm.tab10(np.linspace(0, 1, len(positions)))
        for i, pos in enumerate(positions):
            edge = 'green' if in_room[i] else 'blue' if in_center[i] else 'orange'
            ax.scatter(pos[0], pos[1], c=[agent_colors[i]], s=200, edgecolors=edge, linewidths=3, zorder=5)
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)

        ax.scatter([], [], c='white', s=80, edgecolors='green', linewidths=2, label='In Room')
        ax.scatter([], [], c='white', s=80, edgecolors='blue', linewidths=2, label='In Center')
        ax.scatter([], [], c='white', s=80, edgecolors='orange', linewidths=2, label='Outside')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Info box
        info_text = f"Agents: {env.n_agents} | Rooms: {env.n_rooms} | Capacity: {env.room_capacity}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(image[..., :3])
        plt.close(fig)

        if td["done"].any() or td["terminated"].any():
            td = env.reset()
            obs = td["observation"].to(device)
        else:
            obs = td["observation"].to(device)

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"  GIF saved: {gif_path}")


def train_and_visualize_communication():
    """Train and visualize communication experiments with curriculum."""
    print("\n" + "="*60)
    print("COMMUNICATION EXPERIMENTS (Curriculum Learning)")
    print("="*60)

    exp_dir = EXPERIMENTS_DIR / "communication"
    (exp_dir / "gifs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)

    # Baseline with curriculum
    print("\n--- Baseline (no communication) ---")
    trainer = CurriculumPPOTrainer(TRAIN_CONFIG, DEVICE, name="baseline")
    trainer.train_curriculum(CURRICULUM)
    trainer.save(str(exp_dir / "models" / "baseline.pt"))

    env_viz = MingleEnv(**FINAL_CONFIG, max_steps=200)
    make_gif_with_policy(env_viz, trainer.policy, steps=200,
                        gif_path=str(exp_dir / "gifs" / "baseline_trained.gif"),
                        title="Baseline (6 agents, 3 rooms)")

    # Discrete Communication with curriculum
    print("\n--- Discrete Communication ---")
    try:
        from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
        trainer_comm = CurriculumPPOTrainer(TRAIN_CONFIG, DEVICE, name="discrete_comm")
        trainer_comm.train_curriculum(CURRICULUM)
        trainer_comm.save(str(exp_dir / "models" / "discrete_comm.pt"))

        env_comm_viz = MingleEnvWithComm(**FINAL_CONFIG, max_steps=200, vocab_size=2)
        make_gif_with_policy(env_comm_viz, trainer_comm.policy, steps=200,
                            gif_path=str(exp_dir / "gifs" / "discrete_comm_trained.gif"),
                            title="Discrete Comm (6 agents, 3 rooms)")
    except Exception as e:
        print(f"  Warning: Communication training failed: {e}")


def train_and_visualize_fairness():
    """Train and visualize fairness experiments with curriculum."""
    print("\n" + "="*60)
    print("FAIRNESS EXPERIMENTS (Curriculum Learning)")
    print("="*60)

    exp_dir = EXPERIMENTS_DIR / "fairness"
    (exp_dir / "gifs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)

    fairness_modes = [
        ("baseline", "Baseline"),
        ("gini", "Gini Fairness"),
        ("participation", "Participation"),
    ]

    for name, title in fairness_modes:
        print(f"\n--- {title} ---")
        trainer = CurriculumPPOTrainer(TRAIN_CONFIG, DEVICE, name=name)
        trainer.train_curriculum(CURRICULUM)
        trainer.save(str(exp_dir / "models" / f"{name}.pt"))

        env_viz = MingleEnv(**FINAL_CONFIG, max_steps=200)
        make_gif_with_policy(env_viz, trainer.policy, steps=200,
                            gif_path=str(exp_dir / "gifs" / f"{name}_trained.gif"),
                            title=f"{title} (6 agents, 3 rooms)")


def train_and_visualize_algorithms():
    """Train and visualize algorithm experiments with curriculum."""
    print("\n" + "="*60)
    print("ALGORITHM EXPERIMENTS (Curriculum Learning)")
    print("="*60)

    exp_dir = EXPERIMENTS_DIR / "algorithms"
    (exp_dir / "gifs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)

    algorithms = ["ppo", "ippo", "mappo"]

    for algo in algorithms:
        print(f"\n--- {algo.upper()} ---")
        trainer = CurriculumPPOTrainer(TRAIN_CONFIG, DEVICE, name=algo)
        trainer.train_curriculum(CURRICULUM)
        trainer.save(str(exp_dir / "models" / f"{algo}.pt"))

        env_viz = MingleEnv(**FINAL_CONFIG, max_steps=200)
        make_gif_with_policy(env_viz, trainer.policy, steps=200,
                            gif_path=str(exp_dir / "gifs" / f"{algo}_trained.gif"),
                            title=f"{algo.upper()} (6 agents, 3 rooms)")


def main():
    print("="*60)
    print("CURRICULUM LEARNING: TRAIN AND VISUALIZE")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"\nCurriculum Stages:")
    for stage in CURRICULUM:
        print(f"  - {stage['name'].upper()}: {stage['n_agents']} agents, {stage['n_rooms']} rooms, {stage['frames']} frames")
    print(f"\nFinal Config: {FINAL_CONFIG['n_agents']} agents, {FINAL_CONFIG['n_rooms']} rooms")

    train_and_visualize_communication()
    train_and_visualize_fairness()
    train_and_visualize_algorithms()

    print("\n" + "="*60)
    print("ALL TRAINING AND VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  experiments/communication/models/*.pt")
    print("  experiments/communication/gifs/*_trained.gif")
    print("  experiments/fairness/models/*.pt")
    print("  experiments/fairness/gifs/*_trained.gif")
    print("  experiments/algorithms/models/*.pt")
    print("  experiments/algorithms/gifs/*_trained.gif")


if __name__ == "__main__":
    main()
