"""
Cooperative Mingle - Training Pipeline

Complete training system for the Cooperative Mingle Multi-Agent RL task.
All features from Assignment 2 are integrated:
- Communication Layer (discrete & continuous messaging)
- Fairness Objectives (Gini, participation variance, exclusion counts)
- PPO Training with CTDE (Centralized Training, Decentralized Execution)
- Comprehensive Evaluation with plots and GIFs

Usage:
    python -m src.train.pipeline
"""

import os
import time
import json
from collections import defaultdict
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from src.eval.pipeline import evaluate
from src.eval.gif import make_gif, make_gif_char
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    InsideCenterReward,
    StayInRoomReward,
)
from src.envs.modules.reward_manager import RewardManager, select_reward_modules
from src.train.components import build_train_components
from src.utils.config import load_and_merge_configs


def train(
    collector,
    loss_module,
    advantage_module,
    replay_buffer,
    optim,
    device,
    total_frames,
    frames_per_batch,
    num_epochs,
    minibatch_size,
    max_grad_norm,
    env,
    log_interval=1,
    scheduler=None,
    metrics_save_path="metrics.json",
    gif_interval=None,
    gif_dir="gifs",
    policy_module=None,
    eval_episodes=10,
    use_character_animation=True,
):
    """
    Main training loop with all Assignment 2 features:
    - PPO with entropy annealing and KL early stopping
    - Fairness-aware reward shaping (configured via configs)
    - Communication support (via environment observations)
    - Comprehensive logging and evaluation
    """
    os.makedirs(gif_dir, exist_ok=True)

    logs = defaultdict(list)
    frames_collected = 0
    batch_count = 0
    start_time = time.time()

    # Entropy annealing setup
    total_batches = total_frames // frames_per_batch
    initial_entropy_coef = 0.01
    final_entropy_coef = 0.001

    if hasattr(loss_module, 'set_entropy_coef'):
        print(f"[ENTROPY] Annealing: {initial_entropy_coef} -> {final_entropy_coef}")

    # Character animation paths
    runner_image_paths = [f"visual_utils/squidgame_runner/r{i}.png" for i in range(1, 7)]
    standing_image_paths = [f"visual_utils/squidgame_stand/s{i}.png" for i in range(1, 3)]

    # Generate initial GIF (untrained policy)
    gif_path = os.path.join(gif_dir, "untrained.gif")
    print("[GIF] Generating GIF for untrained model...")
    _generate_gif(env, policy_module, gif_path, use_character_animation,
                  runner_image_paths, standing_image_paths, device)

    # KL early stopping threshold for multi-agent setting
    kl_target = 0.5

    print("\n[TRAINING] Starting training loop...")
    print("=" * 60)

    # Get fairness transform if available
    fairness_transform = getattr(env, 'fairness_transform', None)

    while frames_collected < total_frames:
        for tensordict_data in collector:
            batch_count += 1
            frames_collected += tensordict_data.numel()

            # Apply fairness transform to rewards if enabled
            if fairness_transform is not None:
                tensordict_data = fairness_transform(tensordict_data)

            # Entropy annealing
            if hasattr(loss_module, 'set_entropy_coef'):
                progress = min(1.0, batch_count / max(1, total_batches))
                current_entropy_coef = initial_entropy_coef + progress * (final_entropy_coef - initial_entropy_coef)
                loss_module.set_entropy_coef(current_entropy_coef)

            # Training epochs
            for epoch_idx in range(num_epochs):
                advantage_module(tensordict_data)
                replay_buffer.extend(tensordict_data.cpu())

                num_subbatches = frames_per_batch // minibatch_size
                invalid_count = 0
                total_subbatches = 0
                epoch_losses = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "kl": 0.0}

                for _ in range(num_subbatches):
                    subdata = replay_buffer.sample(minibatch_size).to(device)
                    loss_vals = loss_module(subdata)

                    epoch_losses["actor"] += loss_vals.get("loss_objective", torch.tensor(0.0)).item()
                    epoch_losses["critic"] += loss_vals.get("loss_critic", torch.tensor(0.0)).item()
                    epoch_losses["entropy"] += loss_vals.get("loss_entropy", torch.tensor(0.0)).item()
                    total_subbatches += 1

                    if hasattr(loss_module, 'last_diagnostics') and loss_module.last_diagnostics:
                        epoch_losses["kl"] += loss_module.last_diagnostics.get('kl_approx', 0.0)

                    # Check for invalid losses
                    if any((torch.isnan(val).any() if val.numel() > 0 else False) or
                           (torch.isinf(val).any() if val.numel() > 0 else False)
                           for val in loss_vals.values()):
                        invalid_count += 1
                        optim.zero_grad()
                        continue

                    total_loss = sum(val.mean() if val.numel() > 1 else val for val in loss_vals.values())
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()

                # KL early stopping
                avg_kl = epoch_losses["kl"] / max(1, total_subbatches)
                if avg_kl > kl_target * 1.5:
                    break

            # Update learning rate
            if scheduler is not None:
                before_lr = scheduler.optimizer.param_groups[0]['lr']
                scheduler.step()
                after_lr = scheduler.optimizer.param_groups[0]['lr']
                if after_lr != before_lr:
                    print(f"[LR] {before_lr:.6f} -> {after_lr:.6f}")

            # Log metrics
            reward_mean = tensordict_data["next", "reward"].mean().item()
            logs["reward"].append(reward_mean)
            logs["frames"].append(frames_collected)
            logs["time_elapsed"].append(time.time() - start_time)
            logs["lr"].append(optim.param_groups[0]["lr"])
            logs["actor_loss"].append(epoch_losses["actor"] / max(1, total_subbatches))
            logs["critic_loss"].append(epoch_losses["critic"] / max(1, total_subbatches))
            logs["entropy_loss"].append(epoch_losses["entropy"] / max(1, total_subbatches))

            # Update reward managers
            if env.reward_managers:
                for manager in env.reward_managers.values():
                    manager.update(reward_mean=reward_mean)

            # Print progress
            if batch_count % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Batch {batch_count}] Frames: {frames_collected}/{total_frames} | "
                    f"Reward: {reward_mean:.4f} | "
                    f"Loss: {epoch_losses['actor']/max(1,total_subbatches):.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Generate periodic GIFs
            if gif_interval and batch_count % gif_interval == 0 and policy_module:
                gif_path = os.path.join(gif_dir, f"batch_{batch_count}.gif")
                _generate_gif(env, policy_module, gif_path, use_character_animation,
                             runner_image_paths, standing_image_paths, device)

            if frames_collected >= total_frames:
                break

    print("=" * 60)
    print("[TRAINING] Training complete!\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics JSON
    with open(metrics_save_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in logs.items()}, f, indent=2)
    print(f"[SAVE] Metrics saved to {metrics_save_path}")

    # Generate plots
    plot_dir = os.path.join("plots", f"training_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)

    for metric in ["reward", "actor_loss", "critic_loss", "lr"]:
        if metric in logs and logs[metric]:
            plt.figure(figsize=(10, 6))
            plt.plot(logs["frames"], logs[metric])
            plt.xlabel("Frames")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"Training: {metric.replace('_', ' ').title()}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{metric}.png"))
            plt.close()
    print(f"[SAVE] Plots saved to {plot_dir}")

    # Final GIF
    gif_path = os.path.join(gif_dir, "trained.gif")
    print("[GIF] Generating final trained model GIF...")
    _generate_gif(env, policy_module, gif_path, use_character_animation,
                 runner_image_paths, standing_image_paths, device)

    # Run evaluation
    if policy_module:
        print("\n[EVALUATION] Running evaluation...")
        eval_out_dir = os.path.join("eval_results", f"eval_{timestamp}")
        evaluate(policy_module, env, device, num_episodes=eval_episodes,
                max_steps=env.max_steps, out_dir=eval_out_dir)

    return logs


def _generate_gif(env, policy_module, gif_path, use_character_animation,
                  runner_paths, standing_paths, device):
    """Generate a GIF visualization."""
    try:
        if use_character_animation:
            make_gif_char(env, policy_module, gif_path=gif_path,
                         runner_image_paths=runner_paths,
                         standing_image_paths=standing_paths,
                         use_character_animation=True)
        else:
            make_gif(env, policy_module, gif_path=gif_path,
                    steps=env.max_steps, device=device)
        print(f"[GIF] Saved: {gif_path}")
    except Exception as e:
        print(f"[WARN] GIF failed: {e}")
        try:
            make_gif(env, policy_module, gif_path=gif_path,
                    steps=env.max_steps, device=device)
        except:
            pass


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  Cooperative Mingle - Multi-Agent Reinforcement Learning")
    print("  Assignment 2: Communication, Fairness, Coordination")
    print("=" * 60 + "\n")

    # Load configuration
    config_folder = "configs/"
    print(f"[CONFIG] Loading from: {config_folder}")
    config = load_and_merge_configs(config_folder)
    print("[OK] Configs loaded\n")

    # Show active features
    fairness_cfg = config.get("fairness", {"mode": "none"})
    fairness_mode = fairness_cfg.get("mode", "none")
    if fairness_mode != "none":
        print(f"[FAIRNESS] Active - Mode: {fairness_mode}, Alpha: {fairness_cfg.get('alpha', 0.5)}")
    else:
        print("[FAIRNESS] Disabled (mode: none)")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}\n")

    # Build components
    print("[BUILD] Building training components...")

    reward_modules = [
        InsideCenterReward(inside_reward=10.0, outside_penalty=100.0, phase_mode="spinning"),
        CollisionAvoidanceReward(min_distance=0.5, penalty=5.0, phase_mode="spinning"),
        StayInRoomReward(max_reward=10.0, outside_penalty=20.0, overfill_penalty=20.0, phase_mode="claiming"),
    ]
    for module in reward_modules:
        module._activate()

    components = build_train_components(config, device, reward_modules=reward_modules)
    print("[OK] Components ready\n")

    # Train
    train(
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
        metrics_save_path=config["train"].get("metrics_save_path", "train_metrics.json"),
        gif_interval=10,
        gif_dir="gifs",
        policy_module=components["policy"],
        eval_episodes=config["train"].get("eval_episodes", 10),
        use_character_animation=True,
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
