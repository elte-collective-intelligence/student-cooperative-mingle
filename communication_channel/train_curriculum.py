"""
Curriculum Training Script for Communication-Enabled Agents.

This script implements a 3-stage curriculum:
- Stage 1: 4 agents, 4 rooms (easy - more rooms than needed)
- Stage 2: 4 agents, 3 rooms (medium - must coordinate)
- Stage 3: 6 agents, 3 rooms (hard - full coordination required)

Key improvements:
- Agent IDs for symmetry breaking
- Differentiable communication
- Dense phase-specific rewards
- KL-based early stopping with proper threshold
- Centralized critic for better value estimation
"""

import torch
import os
import sys
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import TransformedEnv, Compose, ObservationNorm

from src.train.modules.gae import StableGAE
from src.utils.config import load_and_merge_configs
from src.eval.gif import make_gif_char
from src.models.critic_factory import build_critic

from communication_channel.envs.discrete_comm_env import MingleEnvWithComm
from communication_channel.models.discrete_comm_policy import build_discrete_comm_policy
from communication_channel.models.multi_action_ppo_loss import MultiActionPPOLoss

# Import reward modules
from src.envs.modules.reward_module import (
    InsideCenterReward,
    CollisionAvoidanceReward,
    StayInRoomReward,
    RoomSpreadReward,
    LeaveToRoomReward,
    RewardModule
)


class CommunicationCoordinationReward(RewardModule):
    """Reward for meaningful communication patterns."""

    def __init__(self, coordination_reward: float = 0.5, phase_mode: str = "both"):
        super().__init__(phase_mode=phase_mode)
        self.coordination_reward = coordination_reward

    def _reward(self, env) -> torch.Tensor:
        n_agents = env.n_agents
        rewards = torch.zeros(n_agents, device=env.device)

        messages = env.message_buffer
        positions = env.agent_positions

        # Reward message diversity
        unique_messages = len(torch.unique(messages))
        diversity_bonus = (unique_messages / env.vocab_size) * 0.1
        rewards += diversity_bonus

        if env.phase == "claiming" and hasattr(env, "room_positions"):
            room_positions = env.room_positions
            agent_room_dists = torch.cdist(positions, room_positions)
            closest_rooms = agent_room_dists.argmin(dim=1)

            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    if closest_rooms[i] != closest_rooms[j] and messages[i] != messages[j]:
                        rewards[i] += self.coordination_reward * 0.5
                        rewards[j] += self.coordination_reward * 0.5
                    elif closest_rooms[i] == closest_rooms[j] and messages[i] == messages[j]:
                        rewards[i] += self.coordination_reward
                        rewards[j] += self.coordination_reward

        return rewards.unsqueeze(1)


# Curriculum stages
CURRICULUM_STAGES = [
    {"name": "Stage 1 (Easy)", "n_agents": 4, "n_rooms": 4, "frames": 10000},
    {"name": "Stage 2 (Medium)", "n_agents": 4, "n_rooms": 3, "frames": 15000},
    {"name": "Stage 3 (Hard)", "n_agents": 6, "n_rooms": 3, "frames": 25000},
]


def create_reward_modules():
    """Create reward modules with phase-specific activations."""
    modules = [
        # Spinning phase rewards
        InsideCenterReward(inside_reward=1.0, outside_penalty=1.0, phase_mode="spinning"),
        CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="spinning"),
        # Claiming phase rewards
        StayInRoomReward(max_reward=1.5, outside_penalty=0.5, overfill_penalty=2.0, phase_mode="claiming"),
        LeaveToRoomReward(leave_reward=0.3, phase_mode="claiming"),
        RoomSpreadReward(spread_bonus=0.5, phase_mode="claiming"),
        # Communication reward (both phases)
        CommunicationCoordinationReward(coordination_reward=0.3, phase_mode="both"),
    ]
    for m in modules:
        m._activate()
    return modules


def build_stage_env(n_agents: int, n_rooms: int, config: dict, device: torch.device, vocab_size: int = 8):
    """Build environment for a curriculum stage."""
    reward_modules = create_reward_modules()

    base_env = MingleEnvWithComm(
        n_agents=n_agents,
        n_rooms=n_rooms,
        arena_radius=config["env"]["arena_radius"],
        center_radius=config["env"]["center_radius"],
        max_steps=config["env"]["max_steps"],
        room_radius=config["env"]["room_radius"],
        room_capacity=2,  # Always 2 per room
        reward_modules=reward_modules,
        vocab_size=vocab_size,
        comm_range=None,  # Global broadcast
        include_agent_id=True,  # Symmetry breaking
    )

    env = TransformedEnv(
        base_env,
        Compose(ObservationNorm(in_keys=["observation"])),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    env.to(device)

    return base_env, env


def train_stage(
    stage_idx: int,
    stage_config: dict,
    config: dict,
    device: torch.device,
    policy=None,
    critic=None,
    output_dir: str = "curriculum_results",
    vocab_size: int = 8,
    prev_n_agents: int = None,
):
    """Train a single curriculum stage."""
    stage_name = stage_config["name"]
    n_agents = stage_config["n_agents"]
    n_rooms = stage_config["n_rooms"]
    total_frames = stage_config["frames"]

    print(f"\n{'='*60}")
    print(f"  {stage_name}: {n_agents} agents, {n_rooms} rooms")
    print(f"  Training for {total_frames} frames")
    print(f"{'='*60}\n")

    # Build environment
    base_env, env = build_stage_env(n_agents, n_rooms, config, device, vocab_size)

    # Check if n_agents changed - must rebuild networks
    if prev_n_agents is not None and prev_n_agents != n_agents:
        print(f"[REBUILD] n_agents changed from {prev_n_agents} to {n_agents}, creating new networks")
        policy = None
        critic = None

    # Build or reuse policy
    if policy is None:
        policy = build_discrete_comm_policy(env, config["policy"], device)
        print("[NEW] Created new policy network")
    else:
        print("[REUSE] Continuing with existing policy weights")

    # Build critic with increased capacity
    if critic is None:
        # Use wider critic for better value estimation
        critic_config = config["critic"].copy()
        critic_config["num_cells"] = critic_config.get("num_cells", 128) * 2  # Double width
        critic = build_critic(env, critic_config, device)
        print("[NEW] Created new critic (increased capacity)")
    else:
        print("[REUSE] Continuing with existing critic weights")

    # Collector
    frames_per_batch = config["train"]["frames_per_batch"]
    collector = SyncDataCollector(
        env, policy, device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        reset_at_each_iter=True
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config["train"]["minibatch_size"]
    )

    # GAE module
    advantage_module = StableGAE(
        gamma=config["ppo"]["gamma"],
        lmbda=config["ppo"]["lambda"],
        value_network=critic,
        average_gae=True,
    )

    # PPO loss with proper settings
    loss_module = MultiActionPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coeff=0.5,
        normalize_advantage=False,
        kl_threshold=0.04,  # Proper KL threshold
    )

    # Optimizer
    lr = config["train"]["lr"]
    optimizer = torch.optim.AdamW(loss_module.parameters(), lr=lr)

    # Training loop
    logs = defaultdict(list)
    frames_collected = 0
    batch_count = 0
    start_time = time.time()
    num_epochs = config["train"]["num_epochs"]
    minibatch_size = config["train"]["minibatch_size"]
    max_grad_norm = config["train"]["max_grad_norm"]

    # Entropy annealing
    total_batches = total_frames // frames_per_batch
    initial_entropy = 0.01
    final_entropy = 0.001

    # GIF settings
    gif_dir = os.path.join(output_dir, f"gifs_stage{stage_idx+1}")
    os.makedirs(gif_dir, exist_ok=True)
    runner_images = [f"visual_utils/squidgame_runner/r{i}.png" for i in range(1, 7)]
    standing_images = [f"visual_utils/squidgame_stand/s{i}.png" for i in range(1, 3)]

    # Generate untrained GIF
    gif_path = os.path.join(gif_dir, f"stage{stage_idx+1}_untrained.gif")
    make_gif_char(base_env, policy, gif_path=gif_path,
                  runner_image_paths=runner_images, standing_image_paths=standing_images,
                  use_character_animation=True)
    print(f"[GIF] Saved untrained GIF: {gif_path}")

    while frames_collected < total_frames:
        for tensordict_data in collector:
            batch_count += 1
            frames_collected += tensordict_data.numel()

            # Entropy annealing
            progress = min(1.0, batch_count / max(1, total_batches))
            current_entropy = initial_entropy + progress * (final_entropy - initial_entropy)
            loss_module.set_entropy_coef(current_entropy)

            for epoch_idx in range(num_epochs):
                advantage_module(tensordict_data)
                replay_buffer.extend(tensordict_data.cpu())

                num_subbatches = frames_per_batch // minibatch_size
                epoch_losses = {"actor": 0, "critic": 0, "entropy": 0}

                for _ in range(num_subbatches):
                    subdata = replay_buffer.sample(minibatch_size).to(device)
                    loss_vals = loss_module(subdata)

                    total_loss = sum(v.mean() if v.numel() > 1 else v for v in loss_vals.values())
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_losses["actor"] += loss_vals.get("loss_objective", torch.tensor(0.0)).item()
                    epoch_losses["critic"] += loss_vals.get("loss_critic", torch.tensor(0.0)).item()
                    epoch_losses["entropy"] += loss_vals.get("loss_entropy", torch.tensor(0.0)).item()

                # KL-based early stopping
                if loss_module.should_stop_early():
                    break

            # Log metrics
            reward_mean = tensordict_data["next", "reward"].mean().item()
            logs["reward"].append(reward_mean)
            logs["frames"].append(frames_collected)
            logs["actor_loss"].append(epoch_losses["actor"] / num_subbatches)
            logs["critic_loss"].append(epoch_losses["critic"] / num_subbatches)
            logs["entropy_loss"].append(epoch_losses["entropy"] / num_subbatches)

            diag = loss_module.last_diagnostics
            logs["entropy"].append(diag.get("entropy", 0))
            logs["kl_approx"].append(diag.get("kl_approx", 0))
            logs["ratio_mean"].append(diag.get("ratio_mean", 1))

            if batch_count % 5 == 0:
                print(f"[Batch {batch_count}] Frames: {frames_collected}/{total_frames} | "
                      f"Reward: {reward_mean:.3f} | KL: {diag.get('kl_approx', 0):.4f} | "
                      f"Entropy: {diag.get('entropy', 0):.2f}")

            # Save periodic GIFs
            if batch_count % 20 == 0:
                gif_path = os.path.join(gif_dir, f"stage{stage_idx+1}_batch{batch_count}.gif")
                make_gif_char(base_env, policy, gif_path=gif_path,
                              runner_image_paths=runner_images, standing_image_paths=standing_images,
                              use_character_animation=True)
                print(f"[GIF] Saved: {gif_path}")

    # Final GIF
    gif_path = os.path.join(gif_dir, f"stage{stage_idx+1}_final.gif")
    make_gif_char(base_env, policy, gif_path=gif_path,
                  runner_image_paths=runner_images, standing_image_paths=standing_images,
                  use_character_animation=True)
    print(f"[GIF] Saved final GIF: {gif_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, f"metrics_stage{stage_idx+1}.json")
    with open(metrics_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in logs.items()}, f, indent=2)
    print(f"[METRICS] Saved: {metrics_path}")

    # Plot metrics
    plot_dir = os.path.join(output_dir, f"plots_stage{stage_idx+1}")
    os.makedirs(plot_dir, exist_ok=True)

    for metric in ["reward", "actor_loss", "critic_loss", "entropy", "kl_approx"]:
        if metric in logs and logs[metric]:
            plt.figure(figsize=(10, 6))
            plt.plot(logs["frames"], logs[metric])
            plt.xlabel("Frames")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{stage_name}: {metric.replace('_', ' ').title()}")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f"{metric}.png"))
            plt.close()

    elapsed = time.time() - start_time
    print(f"\n[DONE] {stage_name} completed in {elapsed:.1f}s")
    print(f"  Final reward: {logs['reward'][-1]:.3f}")
    print(f"  Final entropy: {logs['entropy'][-1]:.3f}")

    return policy, critic, logs, n_agents


def main():
    print("\n" + "="*60)
    print("  CURRICULUM TRAINING WITH COMMUNICATION")
    print("="*60 + "\n")

    # Load config
    config_folder = "configs/"
    config = load_and_merge_configs(config_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"curriculum_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Train through curriculum stages
    policy = None
    critic = None
    all_logs = {}
    prev_n_agents = None

    for stage_idx, stage_config in enumerate(CURRICULUM_STAGES):
        policy, critic, logs, prev_n_agents = train_stage(
            stage_idx=stage_idx,
            stage_config=stage_config,
            config=config,
            device=device,
            policy=policy,
            critic=critic,
            output_dir=output_dir,
            prev_n_agents=prev_n_agents,
        )
        all_logs[f"stage{stage_idx+1}"] = logs

    # Save combined metrics
    combined_path = os.path.join(output_dir, "metrics_all_stages.json")
    with open(combined_path, "w") as f:
        json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                   for k, v in all_logs.items()}, f, indent=2)

    # Create comparison plots
    plt.figure(figsize=(12, 8))
    for stage_name, logs in all_logs.items():
        plt.plot(logs["reward"], label=stage_name)
    plt.xlabel("Batch")
    plt.ylabel("Reward")
    plt.title("Reward Across Curriculum Stages")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"))
    plt.close()

    print("\n" + "="*60)
    print("  CURRICULUM TRAINING COMPLETED")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
