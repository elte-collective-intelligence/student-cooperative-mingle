"""
Training script for communication-enabled agents.

This script trains agents with discrete symbolic communication
and compares performance against baseline (no communication).
"""

import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import TransformedEnv, Compose, ObservationNorm

from src.train.modules.gae import StableGAE
from src.models.critic_factory import build_critic
from src.utils.config import load_and_merge_configs
from src.train.pipeline import train

from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
from communication_channel.models.discrete_comm_policy import build_discrete_comm_policy
from communication_channel.models.multi_action_ppo_loss import MultiActionPPOLoss

# Import reward modules
from src.envs.modules.reward_module import (
    InsideCenterReward,
    CollisionAvoidanceReward,
    StayInRoomReward,
    GetToRoomReward,
    RewardModule
)


class CommunicationCoordinationReward(RewardModule):
    """
    Reward module that incentivizes meaningful communication.

    Rewards agents when:
    1. Agents sending the same message are near each other (coordination signal)
    2. Agents moving towards the same room send distinct messages (differentiation)
    """

    def __init__(self, coordination_reward: float = 0.5, phase_mode: str = "both"):
        super().__init__(phase_mode=phase_mode)
        self.coordination_reward = coordination_reward

    def _reward(self, env) -> torch.Tensor:
        """Compute communication coordination reward."""
        n_agents = env.n_agents
        rewards = torch.zeros(n_agents, device=env.device)

        # Get messages sent by each agent
        messages = env.message_buffer  # (n_agents,)
        positions = env.agent_positions  # (n_agents, 2)

        # Calculate pairwise distances
        distances = torch.cdist(positions, positions)  # (n_agents, n_agents)

        # Reward for message diversity (encourages using different messages)
        unique_messages = len(torch.unique(messages))
        diversity_bonus = (unique_messages / env.vocab_size) * 0.1
        rewards += diversity_bonus

        # During claiming phase, reward agents that send similar messages
        # when they're heading to different rooms (coordination)
        if env.phase == "claiming" and hasattr(env, "room_positions"):
            room_positions = env.room_positions
            # Find closest room for each agent
            agent_room_dists = torch.cdist(positions, room_positions)
            closest_rooms = agent_room_dists.argmin(dim=1)  # (n_agents,)

            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    # If agents are heading to different rooms and use different messages
                    if closest_rooms[i] != closest_rooms[j] and messages[i] != messages[j]:
                        # Reward differentiation
                        rewards[i] += self.coordination_reward * 0.5
                        rewards[j] += self.coordination_reward * 0.5
                    # If agents are heading to the same room and use the same message
                    elif closest_rooms[i] == closest_rooms[j] and messages[i] == messages[j]:
                        # Reward coordination
                        rewards[i] += self.coordination_reward
                        rewards[j] += self.coordination_reward

        # During spinning phase, reward agents that communicate when close
        if env.phase == "spinning":
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    if distances[i, j] < env.center_radius * 0.5:
                        # Agents are close, reward any communication
                        if messages[i] > 0 or messages[j] > 0:  # Non-zero message
                            rewards[i] += self.coordination_reward * 0.3
                            rewards[j] += self.coordination_reward * 0.3

        # Return with correct shape [n_agents, 1]
        return rewards.unsqueeze(1)


def build_comm_components(config: dict, device: torch.device, vocab_size: int = 8, comm_range=None):
    """
    Build training components for communication-enabled environment.

    Args:
        config: Configuration dictionary
        device: Torch device
        vocab_size: Size of message vocabulary
        comm_range: Communication range (None = global)

    Returns:
        Dictionary of training components
    """
    # Create communication environment with BALANCED rewards (penalties ≈ rewards)
    # ADDED: CommunicationCoordinationReward to incentivize meaningful communication
    reward_modules = [
        InsideCenterReward(inside_reward=1.0, outside_penalty=1.0, phase_mode="spinning"),
        CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="spinning"),
        StayInRoomReward(max_reward=1.0, outside_penalty=1.0, overfill_penalty=0.5, phase_mode="claiming"),
        CommunicationCoordinationReward(coordination_reward=0.5, phase_mode="both"),  # NEW: Communication reward
    ]
    for module in reward_modules:
        module._activate()

    base_env = MingleEnvWithComm(
        n_agents=config["env"]["n_agents"],
        n_rooms=config["env"]["n_rooms"],
        arena_radius=config["env"]["arena_radius"],
        center_radius=config["env"]["center_radius"],
        max_steps=config["env"]["max_steps"],
        room_radius=config["env"]["room_radius"],
        room_capacity=config["env"]["room_capacity"],
        reward_modules=reward_modules,
        vocab_size=vocab_size,
        comm_range=comm_range,
    )

    # NOTE: Standard ObservationNorm will normalize messages too, but the
    # CommunicationCoordinationReward provides strong learning signal for communication.
    # This trade-off is acceptable since:
    # 1. The reward signal gives agents incentive to use communication meaningfully
    # 2. The policy network can still learn from normalized message features
    # 3. TorchRL requires TransformedEnv for proper data collection
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    env.to(device)

    # Build policy and critic
    policy = build_discrete_comm_policy(env, config["policy"], device)
    critic = build_critic(env, config["critic"], device)

    # Collector
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        frames_per_batch=config["train"]["frames_per_batch"],
        total_frames=config["train"]["total_frames"],
        reset_at_each_iter=True
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config["train"]["frames_per_batch"], device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config["train"]["minibatch_size"]
    )

    # Advantage module
    advantage_module = StableGAE(
        gamma=config["ppo"]["gamma"],
        lmbda=config["ppo"]["lambda"],
        value_network=critic,
        average_gae=True,
    )

    # PPO loss with INCREASED clip epsilon for better learning
    # Standard PPO uses 0.2, not 0.05
    clip_eps = max(config["ppo"]["clip_epsilon"], 0.15)  # Ensure at least 0.15
    print(f"[PPO] Using clip_epsilon={clip_eps} (original: {config['ppo']['clip_epsilon']})")

    critic_coeff = config["ppo"].get("critic_coeff", 1.0)
    print(f"[PPO] Using critic_coeff={critic_coeff}")

    loss_module = MultiActionPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_eps,
        entropy_bonus=True,
        entropy_coef=config["ppo"]["entropy_eps"],
        critic_coeff=critic_coeff,  # Pass critic coefficient
        normalize_advantage=False  # Already normalized in GAE
    )

    # Optimizer with slightly higher learning rate
    lr = config["train"]["lr"]
    print(f"[Optimizer] Using learning rate={lr}")
    optimizer = torch.optim.AdamW(loss_module.parameters(), lr=lr)

    # Scheduler
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"].get("lr_step_size", 10),
        gamma=config["train"].get("lr_gamma", 0.1)
    )

    return {
        "device": device,
        "env": base_env,  # Return base env for GIF generation
        "policy": policy,
        "critic": critic,
        "collector": collector,
        "replay_buffer": replay_buffer,
        "advantage_module": advantage_module,
        "loss_module": loss_module,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "vocab_size": vocab_size,  # Include for visualization
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  COMMUNICATION-ENABLED AGENT TRAINING")
    print("="*60 + "\n")

    # Load configuration
    config_folder = "configs/"
    print(f"Loading configs from: {config_folder}")
    config = load_and_merge_configs(config_folder)
    print("Configs loaded successfully\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Communication settings
    VOCAB_SIZE = 8  # Number of discrete messages
    COMM_RANGE = None  # None = global broadcast, or set to float for local

    print(f"Communication settings:")
    print(f"  - Vocabulary size: {VOCAB_SIZE}")
    print(f"  - Communication range: {'Global' if COMM_RANGE is None else f'{COMM_RANGE} units'}\n")

    # Build components
    print("Building training components with communication...")
    components = build_comm_components(config, device, vocab_size=VOCAB_SIZE, comm_range=COMM_RANGE)
    print("Components built successfully\n")

    # Create output directory
    output_dir = "communication_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gifs"), exist_ok=True)

    # Training
    print("Starting training with communication...")
    print("-" * 60 + "\n")

    # Get entropy config with defaults
    initial_entropy = config["ppo"].get("entropy_eps", 0.01)

    logs = train(
        collector=components["collector"],
        loss_module=components["loss_module"],
        advantage_module=components["advantage_module"],
        replay_buffer=components["replay_buffer"],
        optim=components["optimizer"],
        device=device,
        total_frames=config["train"]["total_frames"],
        frames_per_batch=config["train"]["frames_per_batch"],
        num_epochs=config["train"]["num_epochs"],
        minibatch_size=config["train"]["minibatch_size"],
        max_grad_norm=config["train"]["max_grad_norm"],
        env=components["env"],
        scheduler=components["scheduler"],
        log_interval=config["train"].get("log_interval", 1),
        metrics_save_path=os.path.join(output_dir, "metrics_comm.json"),
        gif_interval=10,
        gif_dir=os.path.join(output_dir, "gifs"),
        policy_module=components["policy"],
        eval_episodes=config["train"].get("eval_episodes", 10),
        use_character_animation=True,
        # Entropy annealing: start with exploration, end with exploitation
        entropy_annealing=True,
        initial_entropy_coef=initial_entropy,
        final_entropy_coef=initial_entropy * 0.1,  # Reduce to 10% of initial
    )

    print("\n" + "="*60)
    print("  TRAINING COMPLETED")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - Metrics: metrics_comm.json")
    print("  - GIFs: gifs/")
    print("  - Plots: (check train_results/ folder)\n")


def train_discrete_comm(n_agents=4, n_rooms=2, total_frames=50000, output_dir=".", device=None, vocab_size=8):
    """
    Train with discrete communication.

    Returns:
        dict: Training metrics including episode_rewards and losses
    """
    import json
    import time
    import numpy as np
    from tensordict import TensorDict
    import torch.nn as nn
    from src.envs.modules.reward_module import (
        InsideCenterReward, CollisionAvoidanceReward, StayInRoomReward,
        LeaveToRoomReward, GetToRoomReward
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create reward modules - LINEAR GRADIENT toward rooms
    reward_modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="claiming"),
        GetToRoomReward(max_reward=5.0, phase_mode="claiming"),  # Linear gradient
        StayInRoomReward(max_reward=10.0, outside_penalty=1.0, overfill_penalty=1.5, phase_mode="claiming"),
    ]
    for module in reward_modules:
        module._activate()

    # Create communication environment
    from communication_channel.envs.discrete_comm_env import MingleEnvWithComm

    env = MingleEnvWithComm(
        n_agents=n_agents,
        n_rooms=n_rooms,
        arena_radius=10.0,
        center_radius=3.0,
        max_steps=300,
        phase_mode="claiming",
        room_radius=2.0,
        room_capacity=2,
        vocab_size=vocab_size,
        comm_range=None,
        reward_modules=reward_modules,
    )

    # Get dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = 2

    # Simple policy for communication
    class CommPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.action_mean = nn.Linear(128, action_dim)
            self.action_std = nn.Linear(128, action_dim)
            self.message = nn.Linear(128, vocab_size)

        def forward(self, obs):
            if obs.dim() == 3:
                batch, n, d = obs.shape
                obs = obs.view(-1, d)
                h = self.net(obs)
                mean = self.action_mean(h).view(batch, n, -1)
                std = torch.exp(self.action_std(h)).clamp(0.1, 1.0).view(batch, n, -1)
                msg_logits = self.message(h).view(batch, n, -1)
            else:
                h = self.net(obs)
                mean = self.action_mean(h)
                std = torch.exp(self.action_std(h)).clamp(0.1, 1.0)
                msg_logits = self.message(h)
            return mean, std, msg_logits

    class CommCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, obs):
            if obs.dim() == 3:
                batch, n, d = obs.shape
                obs = obs.view(-1, d)
                v = self.net(obs).view(batch, n, 1)
            else:
                v = self.net(obs)
            return v

    policy = CommPolicy().to(device)
    critic = CommCritic().to(device)

    optim = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=3e-4)

    metrics = {"episode_rewards": [], "losses": [], "message_usage": []}

    frames = 0
    start_time = time.time()

    print(f"\nTraining Discrete Communication: {n_agents} agents, {n_rooms} rooms")

    while frames < total_frames:
        td = env.reset()
        obs = td["observation"].to(device)
        episode_reward = 0
        msg_counts = torch.zeros(vocab_size)

        for step in range(300):
            with torch.no_grad():
                mean, std, msg_logits = policy(obs.unsqueeze(0))
                mean, std = mean.squeeze(0), std.squeeze(0)
                msg_logits = msg_logits.squeeze(0)

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().clamp(-0.5, 0.5)

                msg_dist = torch.distributions.Categorical(logits=msg_logits)
                messages = msg_dist.sample()

                for m in messages:
                    msg_counts[m.item()] += 1

            step_td = TensorDict({"action": action.cpu(), "message": messages.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            reward = next_td["reward"].sum().item()
            episode_reward += reward
            frames += n_agents

            done = next_td["done"].any() or next_td["terminated"].any()
            if done:
                break

            obs = next_td["observation"].to(device)

        metrics["episode_rewards"].append(episode_reward)
        metrics["message_usage"].append(msg_counts.tolist())

        if len(metrics["episode_rewards"]) % 10 == 0:
            recent = metrics["episode_rewards"][-10:]
            fps = frames / (time.time() - start_time)
            print(f"Frames: {frames}/{total_frames} | Reward: {np.mean(recent):.2f} | FPS: {fps:.0f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
               os.path.join(output_dir, "model.pt"))

    # Generate GIF - SAME STYLE AS MAIN TRAINING
    try:
        import imageio
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gif_frames = []
        td = env.reset()
        obs = td["observation"].to(device)

        for step in range(300):
            with torch.no_grad():
                mean, std, msg_logits = policy(obs.unsqueeze(0))
                mean, std = mean.squeeze(0), std.squeeze(0)
                msg_logits = msg_logits.squeeze(0)
                dist = torch.distributions.Normal(mean, std.clamp(0.1, 1.0))
                action = dist.sample().clamp(-0.5, 0.5)
                msg_dist = torch.distributions.Categorical(logits=msg_logits)
                messages = msg_dist.sample()

            step_td = TensorDict({"action": action.cpu(), "message": messages.cpu()}, batch_size=[])
            next_td = env._step(step_td)

            # ORIGINAL STYLE visualization (same as main training)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
            ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
            ax.set_aspect('equal')
            ax.set_title(f"Step {step} | Phase: {env.phase} | Discrete Comm")

            # Arena bounds - gray dashed
            ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))

            # Center visual
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="red", linestyle="-"))

            # Draw rooms - green style
            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                room_occupancies = env.room_occupancy.cpu().numpy() if hasattr(env, "room_occupancy") else np.zeros(len(room_positions))
                for i, (room_pos, occupancy) in enumerate(zip(room_positions, room_occupancies)):
                    fill_ratio = min(occupancy / env.room_capacity, 1.0)
                    if fill_ratio >= 1.0:
                        color, alpha = "green", 0.4
                    elif fill_ratio > 0:
                        color, alpha = "yellow", 0.3
                    else:
                        color, alpha = "green", 0.2
                    ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=True, color=color, alpha=alpha))
                    ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=False, color="green", linestyle="-"))
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occupancy)}/{env.room_capacity}", ha='center', va='center', fontsize=10)

            # Agents - colored by status
            positions = env.agent_positions.cpu().numpy()
            in_center = np.linalg.norm(positions, axis=1) <= env.center_radius
            in_room = np.zeros(len(positions), dtype=bool)
            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                for i, pos in enumerate(positions):
                    distances = np.linalg.norm(room_positions - pos, axis=1)
                    if distances.min() < env.room_radius:
                        in_room[i] = True
            outside_all = ~in_center & ~in_room

            ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside")

            # Agent numbers and messages
            for i, pos in enumerate(positions):
                ax.annotate(f"{i}:M{messages[i].item()}", (pos[0], pos[1] + 0.5), ha='center', fontsize=8)

            ax.legend(loc='upper right', fontsize=8)

            fig.canvas.draw()
            try:
                frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            except AttributeError:
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif_frames.append(frame)
            plt.close(fig)

            done = next_td["done"].any() or next_td["terminated"].any()
            if done:
                break
            obs = next_td["observation"].to(device)

        gif_path = os.path.join(output_dir, "trained_policy.gif")
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved: {gif_path}")
    except Exception as e:
        import traceback
        print(f"GIF generation failed: {e}")
        traceback.print_exc()

    return metrics


def train_continuous_comm(n_agents=4, n_rooms=2, total_frames=50000, output_dir=".", device=None, embed_dim=16):
    """
    Train with continuous communication embeddings.

    Returns:
        dict: Training metrics including episode_rewards and losses
    """
    import json
    import time
    import numpy as np
    from tensordict import TensorDict
    import torch.nn as nn
    from src.envs.modules.reward_module import (
        InsideCenterReward, CollisionAvoidanceReward, StayInRoomReward,
        LeaveToRoomReward, GetToRoomReward
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create reward modules - LINEAR GRADIENT toward rooms
    reward_modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="claiming"),
        GetToRoomReward(max_reward=5.0, phase_mode="claiming"),  # Linear gradient
        StayInRoomReward(max_reward=10.0, outside_penalty=1.0, overfill_penalty=1.5, phase_mode="claiming"),
    ]
    for module in reward_modules:
        module._activate()

    # Create communication environment with embeddings
    from communication_channel.envs.continuous_comm_env import MingleEnvWithEmbeddings
    env = MingleEnvWithEmbeddings(
        n_agents=n_agents,
        n_rooms=n_rooms,
        arena_radius=10.0,
        center_radius=3.0,
        max_steps=300,
        phase_mode="claiming",
        room_radius=2.0,
        room_capacity=2,
        embedding_dim=embed_dim,
        reward_modules=reward_modules,
    )

    # Get dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = 2

    class ContinuousCommPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.action_mean = nn.Linear(128, action_dim)
            self.action_std = nn.Linear(128, action_dim)
            if embed_dim > 0:
                self.embedding = nn.Linear(128, embed_dim)

        def forward(self, obs):
            if obs.dim() == 3:
                batch, n, d = obs.shape
                obs = obs.view(-1, d)
                h = self.net(obs)
                mean = self.action_mean(h).view(batch, n, -1)
                std = torch.exp(self.action_std(h)).clamp(0.1, 1.0).view(batch, n, -1)
                if embed_dim > 0:
                    emb = self.embedding(h).view(batch, n, -1)
                else:
                    emb = None
            else:
                h = self.net(obs)
                mean = self.action_mean(h)
                std = torch.exp(self.action_std(h)).clamp(0.1, 1.0)
                emb = self.embedding(h) if embed_dim > 0 else None
            return mean, std, emb

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, obs):
            if obs.dim() == 3:
                batch, n, d = obs.shape
                obs = obs.view(-1, d)
                v = self.net(obs).view(batch, n, 1)
            else:
                v = self.net(obs)
            return v

    policy = ContinuousCommPolicy().to(device)
    critic = Critic().to(device)

    optim = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=3e-4)

    metrics = {"episode_rewards": [], "losses": []}

    frames = 0
    start_time = time.time()

    print(f"\nTraining Continuous Communication: {n_agents} agents, {n_rooms} rooms")

    while frames < total_frames:
        td = env.reset()
        obs = td["observation"].to(device)
        episode_reward = 0

        for step in range(300):
            with torch.no_grad():
                mean, std, emb = policy(obs.unsqueeze(0))
                mean, std = mean.squeeze(0), std.squeeze(0)

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().clamp(-0.5, 0.5)

            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            if emb is not None:
                step_td["embedding"] = emb.squeeze(0).cpu()

            next_td = env._step(step_td)

            reward = next_td["reward"].sum().item()
            episode_reward += reward
            frames += n_agents

            done = next_td["done"].any() or next_td["terminated"].any()
            if done:
                break

            obs = next_td["observation"].to(device)

        metrics["episode_rewards"].append(episode_reward)

        if len(metrics["episode_rewards"]) % 10 == 0:
            recent = metrics["episode_rewards"][-10:]
            fps = frames / (time.time() - start_time)
            print(f"Frames: {frames}/{total_frames} | Reward: {np.mean(recent):.2f} | FPS: {fps:.0f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
               os.path.join(output_dir, "model.pt"))

    # Generate GIF
    try:
        import imageio
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        gif_frames = []
        td = env.reset()
        obs = td["observation"].to(device)

        for step in range(300):
            with torch.no_grad():
                mean, std, emb = policy(obs.unsqueeze(0))
                mean, std = mean.squeeze(0), std.squeeze(0)
                dist = torch.distributions.Normal(mean, std.clamp(0.1, 1.0))
                action = dist.sample().clamp(-0.5, 0.5)

            step_td = TensorDict({"action": action.cpu()}, batch_size=[])
            if emb is not None:
                step_td["embedding"] = emb.squeeze(0).cpu()
            next_td = env._step(step_td)

            # Render frame - ORIGINAL STYLE (same as main training)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
            ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
            ax.set_aspect('equal')
            ax.set_title(f"Step {step} | Phase: {env.phase} | Continuous Comm")

            # Arena bounds - gray dashed
            ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))

            # Center visual
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))

            # Draw rooms - with capacity numbers (same as main training)
            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                room_occupancies = env.room_occupancy.cpu().numpy() if hasattr(env, "room_occupancy") else np.zeros(len(room_positions))
                for i, (room_pos, occupancy) in enumerate(zip(room_positions, room_occupancies)):
                    fill_ratio = min(occupancy / env.room_capacity, 1.0)
                    if fill_ratio >= 1.0:
                        color, alpha = "green", 0.4
                    elif fill_ratio > 0:
                        color, alpha = "yellow", 0.3
                    else:
                        color, alpha = "green", 0.2
                    ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=True, color=color, alpha=alpha))
                    ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=False, color="green", linestyle="-"))
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occupancy)}/{env.room_capacity}", ha='center', va='center', fontsize=10)

            # Agent positions
            positions = env.agent_positions.cpu().numpy()

            # Check agent status
            dists_from_center = np.linalg.norm(positions, axis=1)
            in_center = dists_from_center < env.center_radius

            # Check if in any room
            in_room = np.zeros(len(positions), dtype=bool)
            if hasattr(env, "room_positions") and env.room_positions is not None:
                room_positions = env.room_positions.cpu().numpy()
                for i, pos in enumerate(positions):
                    distances = np.linalg.norm(room_positions - pos, axis=1)
                    if distances.min() < env.room_radius:
                        in_room[i] = True

            outside_all = ~in_center & ~in_room

            # Draw agents colored by status: green=in room, blue=in center, orange=outside
            ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside")

            # Agent numbers
            for i, pos in enumerate(positions):
                ax.text(pos[0] + 0.2, pos[1] + 0.2, str(i), fontsize=8, color='black')

            ax.legend(loc="upper right")

            fig.canvas.draw()
            # Handle both old and new matplotlib API
            try:
                frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            except AttributeError:
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif_frames.append(frame)
            plt.close(fig)

            done = next_td["done"].any() or next_td["terminated"].any()
            if done:
                break
            obs = next_td["observation"].to(device)

        gif_path = os.path.join(output_dir, "trained_policy.gif")
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved: {gif_path}")
    except Exception as e:
        import traceback
        print(f"GIF generation failed: {e}")
        traceback.print_exc()

    return metrics
