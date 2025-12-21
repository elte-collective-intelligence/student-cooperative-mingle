import os
import time
import json
from collections import defaultdict
from datetime import datetime
import torch
import matplotlib.pyplot as plt

from src.eval.pipeline import evaluate
from src.eval.gif import make_gif


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
    metrics_save_path="ippo_metrics.json",
    gif_interval=None,
    gif_dir="gifs_ippo",
    policy_module=None,
    eval_episodes=10,
    use_character_animation=True,  # Added this new argument
):
    """Independent PPO training loop (IPPO)."""

    print("[IPPO] Starting IPPO training...")
    os.makedirs(gif_dir, exist_ok=True)
    logs = defaultdict(list)

    frames_collected = 0
    batch_count = 0
    start_time = time.time()

    # Save a GIF before training
    gif_path = os.path.join(gif_dir, "ippo_untrained.gif")
    make_gif(env, policy_module, gif_path=gif_path, steps=env.max_steps, device=device)

    while frames_collected < total_frames:
        for tensordict_data in collector:
            batch_count += 1
            frames_collected += tensordict_data.numel()

            print(f"[IPPO] Batch {batch_count} - Collected {frames_collected}/{total_frames} frames")

            # Each agent trains independently (shared weights)
            for _ in range(num_epochs):
                advantage_module(tensordict_data)
                replay_buffer.extend(tensordict_data.cpu())

                num_subbatches = frames_per_batch // minibatch_size
                for _ in range(num_subbatches):
                    subdata = replay_buffer.sample(minibatch_size).to(device)
                    loss_vals = loss_module(subdata)

                    total_loss = sum(
                        v.mean() if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
                        for v in loss_vals.values()
                    )

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print("[WARN] Skipping due to invalid loss.")
                        continue

                    optim.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()

            # Logging
            reward_mean = tensordict_data["next", "reward"].mean().item()
            logs["reward"].append(reward_mean)
            logs["frames"].append(frames_collected)
            logs["time_elapsed"].append(time.time() - start_time)
            logs["loss"].append(total_loss.item())

            if batch_count % log_interval == 0:
                print(
                    f"[IPPO Stats] Batch {batch_count} | Reward: {reward_mean:.4f} | "
                    f"Loss: {total_loss.item():.4f}"
                )

            if gif_interval and batch_count % gif_interval == 0:
                gif_path = os.path.join(gif_dir, f"ippo_batch_{batch_count}.gif")
                make_gif(env, policy_module, gif_path=gif_path, steps=env.max_steps, device=device)

            if frames_collected >= total_frames:
                break

    # Save metrics
    with open(metrics_save_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in logs.items()}, f, indent=2)

    # Plot reward curve
    plt.figure()
    plt.plot(logs["frames"], logs["reward"])
    plt.xlabel("Frames")
    plt.ylabel("Mean Reward")
    plt.title("IPPO Training Reward")
    plt.grid(True)
    os.makedirs("train_results", exist_ok=True)
    plt.savefig("train_results/ippo_training_curve.png")
    plt.close()

    print(f"[DONE] IPPO training finished. Metrics saved to {metrics_save_path}")

    # Evaluate
    if policy_module is not None:
        evaluate(policy_module, env, device, num_episodes=eval_episodes, max_steps=env.max_steps)

    return logs
