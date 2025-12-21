import torch
from tqdm import tqdm

from src.eval.gif import make_gif
from src.eval.pipeline import evaluate


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
    scheduler=None,
    log_interval=1,
    metrics_save_path="ppo_metrics.json",
    gif_interval=10,
    policy_module=None,
    eval_episodes=10,
    use_character_animation=True,
    **kwargs
):
    print("🚀 Starting PPO Training...")

    logs = {
        "loss": [],
        "lr": [],
        "frames": [],
        "eval_reward": []   # ⭐ NEW: reward curve storage
    }

    collected_frames = 0
    pbar = tqdm(total=total_frames)

    # --------------------------------------------------------
    # MAIN TRAINING LOOP
    # --------------------------------------------------------
    while collected_frames < total_frames:

        # -------------------------------------------
        # 1. COLLECT ROLLOUTS
        # -------------------------------------------
        for batch in collector:
            batch = batch.to(device)

            replay_buffer.extend(batch)

            collected_frames += batch.numel()
            pbar.update(batch.numel())

            if collected_frames >= frames_per_batch:
                break

        # -------------------------------------------
        # 2. PPO UPDATES
        # -------------------------------------------
        for _ in range(num_epochs):
            for minibatch in replay_buffer.sample(minibatch_size):

                minibatch = minibatch.to(device)

                # ---------- FIX FOR EMPTY BATCH ----------
                if minibatch.batch_size == torch.Size([]):
                    minibatch = minibatch.unsqueeze(0)
                # ------------------------------------------

                loss_vals = loss_module(minibatch)
                loss = loss_vals["loss_objective"]

                optim.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(policy_module.parameters(), max_grad_norm)
                optim.step()

                logs["loss"].append(loss.item())
                logs["frames"].append(collected_frames)
                logs["lr"].append(optim.param_groups[0]["lr"])

            if scheduler:
                scheduler.step()

        # ---------------- FIX ----------------
        replay_buffer.empty()
        # -------------------------------------

        # -------------------------------------------
        # 3. GIF GENERATION
        # -------------------------------------------
        if gif_interval and collected_frames % gif_interval == 0:
            try:
                make_gif(
                    env,
                    policy_module,
                    gif_path=f"gifs_ppo/ppo_step_{collected_frames}.gif",
                    steps=env.max_steps,
                    device=device,
                )
            except Exception as e:
                print(f"⚠️ GIF generation failed: {e}")

        # -------------------------------------------
        # 4. LOGGING
        # -------------------------------------------
        if log_interval and collected_frames % log_interval == 0:
            print(f"[PPO] Frames: {collected_frames} | Loss: {loss.item():.4f}")

        # -------------------------------------------
        # 5. EVALUATION (★ NEW: capture reward)
        # -------------------------------------------
        try:
            eval_result = evaluate(
                policy_module,
                env,
                device,
                num_episodes=eval_episodes,
                max_steps=env.max_steps,
            )

            # ⭐ NEW: store evaluation reward for plotting
            if isinstance(eval_result, dict) and "mean_reward" in eval_result:
                logs["eval_reward"].append(
                    float(eval_result["mean_reward"])
                )
            else:
                # If evaluate() returns a number or tuple
                try:
                    logs["eval_reward"].append(float(eval_result))
                except:
                    logs["eval_reward"].append(None)

        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            logs["eval_reward"].append(None)

    pbar.close()

    # -------------------------------------------
    # SAVE METRICS
    # -------------------------------------------
    import json
    with open(metrics_save_path, "w") as f:
        json.dump(logs, f, indent=4)

    print("🏁 PPO Training complete!")
    print(f"📊 Metrics saved to {metrics_save_path}")

    return logs
