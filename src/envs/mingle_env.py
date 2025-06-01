import torch
import math
from torch import Tensor
from typing import List, Optional, Tuple
from tensordict import TensorDict
from torchrl.envs.common import EnvBase
from torchrl.data import Composite, Bounded, Unbounded, Categorical

from src.envs.modules.reward_manager import RewardManager
from src.envs.modules.reward_module import RewardModule

import torch
import math
from torch import Tensor
from typing import List, Optional, Tuple
from tensordict import TensorDict
from torchrl.envs.common import EnvBase
from torchrl.data import Composite, Bounded, Unbounded, Categorical

from src.envs.modules.reward_manager import RewardManager
from src.envs.modules.reward_module import RewardModule

class MingleEnv(EnvBase):
    """
    MingleEnv simulates an environment where agents move in a circular arena, transition between
    spinning and claiming phases, and interact with rooms.

    Parameters:
        n_agents (int): Number of agents in the environment.
        n_rooms (int): Number of rooms distributed around the arena.
        arena_radius (float): Outer radius of the arena.
        center_radius (float): Radius of the central region used during the spinning phase.
        max_steps (int): Maximum number of steps in an episode.
        spinning_phase_range (tuple): Range (min, max) for random spinning phase duration.
        room_radius (float): Radius of each room (used for positioning).
        room_capacity (int): Max number of agents that can occupy a room.
        reward_modules (List[RewardModule], optional): List of callable modules that provide reward.
        phase_mode (str): Controls phase behavior ("both", "claiming", etc.).
    """
    def __init__(
        self,
        n_agents: int = 2,
        n_rooms: int = 2,
        arena_radius: float = 10.0,
        center_radius: float = 3.0,
        max_steps: int = 300,
        spinning_phase_range: Tuple[int, int] = (50, 100),
        room_radius: float = 3.0,
        room_capacity: int = 2,
        reward_modules: Optional[List[RewardModule]] = None,
        reward_managers: Optional[dict] = None,
        phase_mode: str = "both"
    ) -> None:
        super().__init__()

        self.n_agents = n_agents
        self.n_rooms = n_rooms
        self.arena_radius = arena_radius
        self.center_radius = center_radius
        self.max_steps = max_steps
        self.spinning_phase_range = spinning_phase_range
        self.max_speed = 0.2
        self.room_radius = room_radius
        self.room_capacity = room_capacity
        self.reward_modules = reward_modules or []
        self.reward_managers = reward_managers
        self.phase_mode = phase_mode

        self.agent_positions = torch.zeros(self.n_agents, 2, device=self.device)
        self.room_positions = None
        self.room_occupancy = None

        self.observation_spec = Composite({
            "observation": Unbounded(shape=(self.n_agents, 14), device=self.device),
        })
        self.action_spec = Composite({
            "action": Bounded(
                low=-self.max_speed,
                high=self.max_speed,
                shape=(self.n_agents, 2),
                device=self.device,
            ),
        })
        self.reward_spec = Unbounded(shape=(self.n_agents, 1), device=self.device)
        self.done_spec = Composite({
            "done": Categorical(n=2, shape=(self.n_agents, 1), dtype=torch.bool),
            "terminated": Categorical(n=2, shape=(self.n_agents, 1), dtype=torch.bool),
        }, device=self.device)

    def _init_agents(self) -> None:
        """Randomly initializes agent positions within the center circle."""
        angles = 2 * math.pi * torch.rand(self.n_agents, 1)
        radii = self.center_radius * torch.sqrt(torch.rand(self.n_agents, 1))
        self.agent_positions = torch.cat(
            [radii * torch.cos(angles), radii * torch.sin(angles)], dim=1
        ).to(self.device)

    def _init_rooms(self) -> None:
        """Initializes rooms evenly spaced around the arena circumference, pulled in from the edge."""
        edge_margin = 0.2 * self.room_radius
        angles = torch.linspace(0.0, 2 * math.pi, steps=self.n_rooms + 1)[:-1]

        radius = self.arena_radius - self.room_radius - edge_margin
        self.room_positions = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1).to(self.device)

        self.room_occupancy = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

    def _rotate_positions(self, angle_rad: float) -> None:
        """Rotates agent positions around the center by a given angle."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        rot_matrix = torch.tensor([[c, -s], [s, c]], device=self.device)
        self.agent_positions = self.agent_positions @ rot_matrix.T

    def _enforce_boundaries(self) -> None:
        """Prevents agents from moving outside the arena."""
        dists = self.agent_positions.norm(dim=1)
        outside = dists > self.arena_radius
        if outside.any():
            self.agent_positions[outside] *= (
                self.arena_radius / dists[outside]
            ).unsqueeze(1)

    def _compute_observations(self) -> Tensor:
        """Constructs a 14-dimensional observation vector for each agent."""
        pos = self.agent_positions
        norm = pos.norm(dim=1, keepdim=True)

        dist_to_center = norm / self.center_radius
        direction_to_center = -pos / (norm + 1e-8)
        dist_to_center_edge = (norm - self.center_radius) / self.center_radius

        # Distance to rooms
        room_dists = torch.cdist(pos, self.room_positions)
        closest_room = room_dists.argmin(dim=1)
        dist_to_room = room_dists[torch.arange(self.n_agents), closest_room].unsqueeze(1) / self.arena_radius

        # Direction to closest room
        delta_room = self.room_positions[closest_room] - pos
        nearest_room_dir = delta_room / (delta_room.norm(dim=1, keepdim=True) + 1e-8)

        # Signed distance to edge of the closest room (normalized)
        raw_room_dist = room_dists[torch.arange(self.n_agents), closest_room].unsqueeze(1)
        signed_dist_to_room_edge = (self.room_radius - raw_room_dist) / self.room_radius  # inside = positive

        # Room occupancy computation
        assignments = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        for i in range(self.n_agents):
            close_rooms = (room_dists[i] < self.room_radius).nonzero(as_tuple=True)[0]
            if close_rooms.numel() > 0:
                closest = close_rooms[room_dists[i, close_rooms].argmin()]
                assignments[i] = closest
        self.room_occupancy = torch.bincount(assignments[assignments >= 0], minlength=self.n_rooms)

        capacity_tensor = torch.full((self.n_agents, 1), self.room_capacity, device=self.device)
        occupancy_tensor = self.room_occupancy[closest_room].unsqueeze(1).float() / self.room_capacity

        # Distance to nearest agent
        agent_dists = torch.cdist(pos, pos)
        agent_dists.fill_diagonal_(float("inf"))
        nearest_agent_idx = agent_dists.argmin(dim=1)
        nearest_directions = torch.stack([
            (pos[j] - pos[i]) / ((pos[j] - pos[i]).norm() + 1e-8)
            for i, j in enumerate(nearest_agent_idx)
        ])
        dist_to_agent = agent_dists.min(dim=1, keepdim=True).values / self.arena_radius

        phase_flag = torch.full((self.n_agents, 1), 1.0 if self.phase == "claiming" else 0.0, device=self.device)

        obs = torch.cat([
            dist_to_center.clamp(0, 1),
            direction_to_center,
            dist_to_center_edge,
            dist_to_room.clamp(0, 1),
            nearest_room_dir,
            signed_dist_to_room_edge,
            capacity_tensor,
            occupancy_tensor,
            dist_to_agent.clamp(0, 1),
            nearest_directions,
            phase_flag
        ], dim=1)

        if torch.isnan(obs).any() or torch.isinf(obs).any():
            obs = torch.nan_to_num(obs)

        return obs

    def _compute_rewards(self) -> Tensor:
        """Aggregates rewards from reward modules."""
        rewards = torch.zeros((self.n_agents, 1), device=self.device)
        if self.reward_managers and self.phase in self.reward_managers:
            rewards = self.reward_managers[self.phase](self)
        else:
            rewards = torch.zeros((self.n_agents, 1), device=self.device)
            for module in self.reward_modules:
                rewards += module(self)
        return rewards

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Resets the environment state for a new episode."""
        self._init_agents()
        self._init_rooms()
        self.current_step = 0
        self.spinning_duration = torch.randint(*self.spinning_phase_range, (1,)).item()
        self.phase = "spinning" if self.phase_mode != "claiming" else "claiming"

        obs = self._compute_observations()
        done = torch.zeros((self.n_agents, 1), dtype=torch.bool, device=self.device)

        return TensorDict({
            "observation": obs,
            "done": done.clone(),
            "terminated": torch.full_like(done, False),
        }, batch_size=[])

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Performs one simulation step given agent actions."""
        self.current_step += 1
        actions = tensordict["action"]

        # Normalize actions exceeding max speed
        speeds = actions.norm(dim=1, keepdim=True)
        actions = torch.where(speeds > self.max_speed, actions / speeds * self.max_speed, actions)

        self.agent_positions += actions
        self._enforce_boundaries()

        if self.phase == "spinning" and self.phase_mode == "both" and self.current_step >= self.spinning_duration:
            self.phase = "claiming"

        if self.phase == "spinning":
            in_center = self.agent_positions.norm(dim=1) <= self.center_radius
            if in_center.any():
                self._rotate_positions(0.05)

        obs = self._compute_observations()
        reward = self._compute_rewards()
        done = torch.zeros((self.n_agents, 1), dtype=torch.bool, device=self.device)

        return TensorDict({
            "observation": obs,
            "reward": reward,
            "done": done.clone(),
            "terminated": torch.full_like(done, self.current_step >= self.max_steps),
        }, batch_size=[])

    def _set_seed(self, seed: int) -> int:
        """Sets the seed for torch RNG."""
        torch.manual_seed(seed)
        return seed