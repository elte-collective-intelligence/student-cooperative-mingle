import torch
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.envs.mingle_env import MingleEnv

class RewardModule:
    """
    Abstract base class for all reward modules. 
    Supports phase-based reward filtering: "spinning", "claiming", or "both".
    """
    def __init__(self, phase_mode: str = "both", active: bool = False):
        assert phase_mode in {"spinning", "claiming", "both"}, f"Invalid phase_mode: {phase_mode}"
        self.phase_mode = phase_mode
        self.active = active

    def __call__(self, env: "MingleEnv") -> Tensor:
        if env.phase != self.phase_mode or not self.active:
            return torch.zeros((env.n_agents, 1), device=env.device)
        return self._reward(env)

    def _activate(self):
        self.active = True

    def _deactivate(self):
        self.active = False

    def _reward(self, env: "MingleEnv") -> Tensor:
        raise NotImplementedError("Subclasses must implement the _reward method.")


class CloseToCenterReward(RewardModule):
    def __init__(self, phase_mode: str = "both"):
        super().__init__(phase_mode=phase_mode)

    def _reward(self, env: "MingleEnv") -> Tensor:
        dists = env.agent_positions.norm(dim=1, keepdim=True)
        return 1.0 - dists.clamp(0, env.center_radius) / env.center_radius


class InsideCenterReward(RewardModule):
    def __init__(self, phase_mode: str = "both", inside_reward: float = 1.0, outside_penalty: float = -1.0):
        super().__init__(phase_mode=phase_mode)
        self.inside_reward = inside_reward
        self.outside_penalty = outside_penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        dists = env.agent_positions.norm(dim=1, keepdim=True) / env.center_radius
        inside_mask = (dists <= 1.0).float()
        outside_mask = 1.0 - inside_mask
        return (inside_mask * self.inside_reward) - (outside_mask * self.outside_penalty)


class CollisionAvoidanceReward(RewardModule):
    def __init__(self, phase_mode: str = "both", min_distance: float = 0.5, penalty: float = 1.0):
        super().__init__(phase_mode=phase_mode)
        self.min_distance = min_distance
        self.penalty = penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        dists = torch.cdist(env.agent_positions, env.agent_positions)
        dists.fill_diagonal_(self.min_distance + 1.0)
        too_close = (dists < self.min_distance).any(dim=1, keepdim=True).float()
        return -self.penalty * too_close

class GetToRoomReward(RewardModule):
    def __init__(
        self,
        phase_mode: str = "both",
        max_reward: float = 1.0,
        min_distance: float = 0.0,
        full_room_penalty: float = 1.0,
        epsilon: float = 1e-6,  # Small constant to avoid division by zero
    ):
        super().__init__(phase_mode=phase_mode)
        self.max_reward = max_reward
        self.min_distance = min_distance
        self.full_room_penalty = full_room_penalty
        self.epsilon = epsilon

    def _reward(self, env: "MingleEnv") -> Tensor:
        # Compute distances between agents and rooms
        room_dists = torch.cdist(env.agent_positions, env.room_positions)
        min_dists, closest_room_indices = room_dists.min(dim=1, keepdim=True)  # [N, 1]

        # Get room occupancy info
        closest_room_occupancy = env.room_occupancy[closest_room_indices.squeeze(1)]
        room_capacity = env.room_capacity
        is_full = (closest_room_occupancy >= room_capacity).float().unsqueeze(1)  # [N, 1]

        # Apply quadratic scaling
        shifted_dists = (min_dists - self.min_distance).clamp(min=0.0)
        scaled = 1.0 - (shifted_dists / (min_dists + self.epsilon)).pow(2)

        # Compute rewards and penalties
        reward = self.max_reward * scaled * (1 - is_full)
        penalty = self.full_room_penalty * scaled * is_full

        final_reward = reward - penalty
        return final_reward

class StayInRoomReward(RewardModule):
    def __init__(
        self,
        phase_mode: str = "both",
        max_reward: float = 1.0,
        outside_penalty: float = -1.0,
        overfill_penalty: float = -1.0,
    ):
        super().__init__(phase_mode=phase_mode)
        self.max_reward = max_reward
        self.outside_penalty = outside_penalty
        self.overfill_penalty = overfill_penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        # Compute distances between agents and rooms
        room_dists = torch.cdist(env.agent_positions, env.room_positions)

        # Determine closest room and distance to it
        min_dists, closest_room_indices = room_dists.min(dim=1)

        # Mask of agents inside any room
        in_room_mask = (room_dists < env.room_radius).any(dim=1, keepdim=True).float()
        outside_mask = 1.0 - in_room_mask

        # Get occupancy of closest room for each agent
        closest_room_occupancy = env.room_occupancy[closest_room_indices]
        room_capacity = env.room_capacity

        # Compute fill ratio and scale reward
        fill_ratio = (closest_room_occupancy.float() / room_capacity).clamp(0, 1).unsqueeze(1)
        scaled_reward = self.max_reward * fill_ratio * in_room_mask

        # Overfill penalty
        is_overfilled = (closest_room_occupancy > room_capacity).float().unsqueeze(1)
        overfill_penalty = self.overfill_penalty * in_room_mask * is_overfilled

        # Combine all components
        reward = scaled_reward - outside_mask * self.outside_penalty - overfill_penalty

        return reward

