import time
import json
from collections import defaultdict
import torch

from src.eval.pipeline import evaluate
from src.eval.gif import make_gif, make_gif_char
from src.envs.modules.reward_module import CollisionAvoidanceReward, InsideCenterReward, StayInRoomReward
from src.envs.modules.reward_manager import RewardManager, select_reward_modules
from src.train.components import build_train_components
from src.utils.config import load_and_merge_configs

import time
import torch
import json
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from datetime import datetime


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
    use_character_animation=True
):
    os.makedirs(gif_dir, exist_ok=True)

    logs = defaultdict(list)
    frames_collected = 0
    batch_count = 0
    start_time = time.time()

    gif_path = os.path.join(gif_dir, f"mingle_untrained.gif")
    print(f"🎥 Generating GIF for untrained model...")
    runner_image_paths = [
        "visual_utils/squidgame_runner/r1.png",
        "visual_utils/squidgame_runner/r2.png",
        "visual_utils/squidgame_runner/r3.png",
        "visual_utils/squidgame_runner/r4.png",
        "visual_utils/squidgame_runner/r5.png",
        "visual_utils/squidgame_runner/r6.png",
    ]
    standing_image_paths = [
        "visual_utils/squidgame_stand/s1.png",
        "visual_utils/squidgame_stand/s2.png",
    ]
    if not use_character_animation:
        make_gif(env, policy_module, gif_path=gif_path, steps=env.max_steps, device=device)
    else:
        make_gif_char(env, policy_module, gif_path=gif_path, runner_image_paths=runner_image_paths, standing_image_paths=standing_image_paths, use_character_animation=use_character_animation)

    while frames_collected < total_frames:
        for tensordict_data in collector:
            batch_count += 1
            frames_collected += tensordict_data.numel()

            print(f"📦 Batch {batch_count} - Collected {frames_collected}/{total_frames} frames")

            for _ in range(num_epochs):
                advantage_module(tensordict_data)
                replay_buffer.extend(tensordict_data.cpu())

                num_subbatches = frames_per_batch // minibatch_size
                invalid_count = 0
                total_subbatches = 0

                epoch_actor_loss = 0.0
                epoch_critic_loss = 0.0
                epoch_entropy_loss = 0.0

                for _ in range(num_subbatches):
                    subdata = replay_buffer.sample(minibatch_size).to(device)
                    loss_vals = loss_module(subdata)

                    actor_loss = loss_vals.get("loss_objective", torch.tensor(0.0, device=device)).item()
                    critic_loss = loss_vals.get("loss_critic", torch.tensor(0.0, device=device)).item()
                    entropy_loss = loss_vals.get("loss_entropy", torch.tensor(0.0, device=device)).item()

                    epoch_actor_loss += actor_loss
                    epoch_critic_loss += critic_loss
                    epoch_entropy_loss += entropy_loss
                    total_subbatches += 1

                    if any(torch.isnan(val) or torch.isinf(val) for val in loss_vals.values()):
                        invalid_count += 1
                        optim.zero_grad()
                        continue

                    total_loss = sum(loss_vals.values())
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()

                invalid_ratio = invalid_count / total_subbatches if total_subbatches > 0 else 0.0

            if scheduler is not None:
                before_lr = scheduler.optimizer.param_groups[0]['lr']
                scheduler.step()
                after_lr = scheduler.optimizer.param_groups[0]['lr']

                if after_lr != before_lr:
                    print(f"\n🔄 Learning rate updated: {before_lr:.6f} ➡️  {after_lr:.6f}\n")

            reward_mean = tensordict_data["next", "reward"].mean().item()
            lr_current = optim.param_groups[0]["lr"]

            if env.reward_managers:
                for manager in env.reward_managers.values():
                    manager.update(reward_mean=reward_mean)

            logs["reward"].append(reward_mean)
            logs["lr"].append(lr_current)
            logs["frames"].append(frames_collected)
            logs["time_elapsed"].append(time.time() - start_time)
            logs["actor_loss"].append(epoch_actor_loss / max(1, total_subbatches))
            logs["critic_loss"].append(epoch_critic_loss / max(1, total_subbatches))
            logs["entropy_loss"].append(epoch_entropy_loss / max(1, total_subbatches))
            logs["invalid_ratio"].append(invalid_ratio)

            if batch_count % log_interval == 0 or frames_collected >= total_frames:
                elapsed = time.time() - start_time
                print(
                    f"📊 Stats [Batch {batch_count}]"
                    f"\n    🕒 Time Elapsed: {elapsed:.1f}s"
                    f"\n    🎯 Avg Reward: {reward_mean:.4f}"
                    f"\n    🎭 Actor Loss: {epoch_actor_loss / max(1, total_subbatches):.4f}"
                    f"\n    🏛️  Critic Loss: {epoch_critic_loss / max(1, total_subbatches):.4f}"
                    f"\n    🔥 Entropy Loss: {epoch_entropy_loss / max(1, total_subbatches):.4f}"
                    f"\n    📉 Learning Rate: {lr_current:.6f}"
                    f"\n    ⚠️  Invalid Subbatches: {invalid_count}/{total_subbatches} ({invalid_ratio:.1%})\n"
                )

            if gif_interval is not None and batch_count % gif_interval == 0 and policy_module is not None:
                gif_path = os.path.join(gif_dir, f"mingle_batch_{batch_count}.gif")
                print(f"🎥 Generating GIF for Batch {batch_count}...")
                make_gif(env, policy_module, gif_path=gif_path, steps=env.max_steps)

    # Save training logs
    with open(metrics_save_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in logs.items()}, f, indent=2)

    # Plot training logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join("train_results", f"training_metrics_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)

    def plot_metric(metric_name, ylabel=None):
        plt.figure()
        plt.plot(logs["frames"], logs[metric_name])
        plt.xlabel("Frames")
        plt.ylabel(ylabel or metric_name.replace("_", " ").title())
        plt.title(f"{ylabel or metric_name.replace('_', ' ').title()} Over Time")
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(plot_dir, f"{metric_name}_over_time.png")
        plt.savefig(filename)
        plt.close()

    for metric in ["reward", "lr", "actor_loss", "critic_loss", "entropy_loss", "invalid_ratio"]:
        plot_metric(metric)

    print(f"📈 Saved training plots to {plot_dir}")

    # Run evaluation
    if policy_module is not None:
        print("\n🏁 Training complete.")
        eval_out_dir = os.path.join("eval_results", f"eval_metrics_{timestamp}")
        evaluate(policy_module, env, device, num_episodes=eval_episodes, max_steps=env.max_steps, out_dir=eval_out_dir)
    else:
        print("⚠️ Skipping evaluation: No policy_module provided.")

    return logs


if __name__ == "__main__":
    print("\n🚀 Starting training script\n")

    config_folder = "configs/"
    print(f"📂 Loading configs from folder: {config_folder}")
    config = load_and_merge_configs(config_folder)
    print("✅ Configs loaded and merged successfully\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")

    print("🧱 Building training components...")

    manual_config = input("Do you want to configure rewards manually? (y/n): ").strip().lower()
    if manual_config == "y":
        print("⚙️  Configuring rewards manually...")
        selected_modules, thresholds = select_reward_modules()
        reward_managers = {
            phase: RewardManager(modules, thresholds[phase], phase)
            for phase, modules in selected_modules.items()
        }
        components = build_train_components(config, device, reward_managers=reward_managers)
    else:
        print("⚙️  Using predefined reward modules...")
        reward_modules = [
            InsideCenterReward(inside_reward=10.0, outside_penalty=100.0, phase_mode="spinning"),
            CollisionAvoidanceReward(min_distance=0.5, penalty=5.0, phase_mode="spinning"),
            StayInRoomReward(max_reward=10.0, outside_penalty=20.0, overfill_penalty=20.0, phase_mode="claiming"),
        ]
        for module in reward_modules:
            module._activate()

        components = build_train_components(config, device, reward_modules=reward_modules)

    print("✅ Training components built successfully\n")

    print("🏁 Starting training loop")
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
        metrics_save_path=config["train"].get("metrics_save_path", "train_metrics.json"),
        gif_interval=10,
        policy_module=components["policy"],
        eval_episodes=config["train"].get("eval_episodes", 10),
        use_character_animation=True
    )
    print("✅ Training & Evaluation finished successfully\n")

    metrics_path = config["train"].get("metrics_save_path", "train_metrics.json")
    print(f"💾 Saved training metrics to {metrics_path}\n")