from typing import List, Optional
import torch

from src.envs.dynamic_mingle_env import DynamicMingleEnv
from src.envs.modules.reward_manager import RewardManager
from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import RewardModule

def make_env(env_config: dict, device: torch.device = torch.device("cpu"), 
             reward_modules: Optional[List[RewardModule]] = None, reward_managers: Optional[dict] = None) -> MingleEnv:
    """
    Creates an instance of MingleEnv from configuration.

    Args:
        env_config (dict): Dictionary containing environment configuration.
        device (torch.device): Torch device to run the environment on.

    Returns:
        MingleEnv: Configured environment instance.
    """
    is_dynamic = env_config.get("dynamic", False)
    if is_dynamic:
        env = DynamicMingleEnv(
            n_agents=env_config.get("n_agents", 2),
            n_rooms=env_config.get("n_rooms", 2),
            arena_radius=env_config.get("arena_radius", 10.0),
            center_radius=env_config.get("center_radius", 3.0),
            max_steps=env_config.get("max_steps", 300),
            spinning_phase_range=tuple([env_config.get("spinning_phase_range_start", 50), env_config.get("spinning_phase_range_end", 100)]),
            room_radius=env_config.get("room_radius", 3.0),
            room_capacity=env_config.get("room_capacity", 2),
            reward_modules=reward_modules,
            reward_managers=reward_managers,
            phase_mode=env_config.get("phase_mode", "both")
        )
        env.to(device)
    else:
        env = MingleEnv(
            n_agents=env_config.get("n_agents", 2),
            n_rooms=env_config.get("n_rooms", 2),
            arena_radius=env_config.get("arena_radius", 10.0),
            center_radius=env_config.get("center_radius", 3.0),
            max_steps=env_config.get("max_steps", 300),
            spinning_phase_range=tuple([env_config.get("spinning_phase_range_start", 50), env_config.get("spinning_phase_range_end", 100)]),
            room_radius=env_config.get("room_radius", 3.0),
            room_capacity=env_config.get("room_capacity", 2),
            reward_modules=reward_modules,
            reward_managers=reward_managers,
            phase_mode=env_config.get("phase_mode", "both")
        )
        env.to(device)
    return env
