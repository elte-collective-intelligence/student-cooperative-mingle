from typing import List, Optional
import torch
import multiprocessing
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter, ObservationNorm
import torch.optim.lr_scheduler as lr_scheduler

from src.envs.modules.reward_manager import RewardManager, select_reward_modules
from src.models.policy_factory import build_policy
from src.models.critic_factory import build_critic
from src.envs.env_factory import make_env
from src.envs.modules.reward_module import RewardModule
from src.train.modules.gae import StableGAE
from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform

from torchrl.objectives import ClipPPOLoss


def build_train_components(config: dict, device: torch.device = torch.device("cpu"),
                           reward_modules: Optional[List[RewardModule]] = None, reward_managers: Optional[dict] = None):
    """Build all training components for Cooperative Mingle."""

    # Fairness config
    fairness_cfg = config.get("fairness", {"mode": "none"})
    fairness_mode = fairness_cfg.get("mode", "none")

    if fairness_mode != "none":
        print(f"[FAIRNESS] Enabling fairness transform with mode: {fairness_mode}")
        fairness_transform = make_fairness_reward_transform(fairness_cfg)
    else:
        fairness_transform = None

    # Create base environment
    base_env = make_env(
        env_config=config["env"],
        device=device,
        reward_modules=reward_modules,
        reward_managers=reward_managers
    )

    # Create transformed environment (without ObservationNorm to avoid size issues)
    env = TransformedEnv(base_env)
    env.reward_managers = reward_managers

    # Store fairness transform for use in training loop
    env.fairness_transform = fairness_transform

    env.to(device)

    # Build policy & critic
    policy = build_policy(env, config["policy"], device)
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

    # PPO loss
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=config["ppo"]["clip_epsilon"],
        entropy_bonus=True,
        entropy_coef=config["ppo"]["entropy_eps"]
    )

    # Optimizer
    optimizer = torch.optim.AdamW(loss_module.parameters(), lr=config["train"]["lr"])

    # Scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"].get("lr_step_size", 10),
        gamma=config["train"].get("lr_gamma", 0.1)
    )

    return {
        "device": device,
        "env": env,
        "policy": policy,
        "critic": critic,
        "collector": collector,
        "replay_buffer": replay_buffer,
        "advantage_module": advantage_module,
        "loss_module": loss_module,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
