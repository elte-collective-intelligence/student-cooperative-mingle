import os
import sys
import time
import json
import argparse
from collections import defaultdict
from datetime import datetime

import torch
import matplotlib.pyplot as plt

# ==========================================================
# 🧠 Algorithm Selection (PPO or IPPO)
# ==========================================================
algorithm = None
if "--algorithm" in sys.argv:
    idx = sys.argv.index("--algorithm") + 1
    if idx < len(sys.argv):
        algorithm = sys.argv[idx].lower()

if algorithm == "ippo":
    print("🧠 IPPO mode active.")
    from src.train.ippo_trainer import train
else:
    print("🤖 PPO mode active.")
    from src.train.ppo_trainer import train

# ==========================================================
# 🧩 Core Imports
# ==========================================================
from src.eval.pipeline import evaluate
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    InsideCenterReward,
    StayInRoomReward,
)
from src.envs.modules.reward_manager import RewardManager, select_reward_modules
from src.train.components import build_train_components
from src.utils.config import load_and_merge_configs


# ==========================================================
# 🚀 Main Training Script
# ==========================================================
if __name__ == "__main__":
    print("\n🚀 Starting training script\n")

    # Load configuration
    config_folder = "configs/"
    print(f"📂 Loading configs from folder: {config_folder}")
    config = load_and_merge_configs(config_folder)
    print("✅ Configs loaded and merged successfully\n")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}\n")

    # Build training components
    print("🧱 Building training components...")

    manual_config = input("Do you want to configure rewards manually? (y/n): ").strip().lower()
    if manual_config == "y":
        print("⚙️ Configuring rewards manually...")
        selected_modules, thresholds = select_reward_modules()
        reward_managers = {
            phase: RewardManager(modules, thresholds[phase], phase)
            for phase, modules in selected_modules.items()
        }
        components = build_train_components(
            config,
            device,
            reward_managers=reward_managers,
        )
    else:
        print("⚙️ Using predefined reward modules...")
        reward_modules = [
            InsideCenterReward(inside_reward=10.0, outside_penalty=100.0, phase_mode="spinning"),
            CollisionAvoidanceReward(min_distance=0.5, penalty=5.0, phase_mode="spinning"),
            StayInRoomReward(max_reward=10.0, outside_penalty=20.0, overfill_penalty=20.0, phase_mode="claiming"),
        ]
        for module in reward_modules:
            module._activate()

        components = build_train_components(config, device, reward_modules=reward_modules)

    print("✅ Training components built successfully\n")

    # Run training
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
        use_character_animation=True,
    )

    print("✅ Training & Evaluation finished successfully\n")

    # Save metrics info
    metrics_path = config["train"].get("metrics_save_path", "train_metrics.json")
    print(f"💾 Saved training metrics to {metrics_path}\n")
