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

class DynamicMingleEnv(EnvBase):
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
        max_speed: float = 0.5,
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
        self.max_speed = max_speed
        self.room_radius = room_radius
        self.room_capacity = room_capacity
        self.reward_modules = reward_modules or []
        self.reward_managers = reward_managers
        self.phase_mode = phase_mode

        self.agent_positions = torch.zeros(self.n_agents, 2, device=self.device)
        self.room_positions = None
        self.room_occupancy = None

        self.agent_facing = torch.zeros(self.n_agents, device=self.device)
        self.agent_velocity = torch.zeros(self.n_agents, 2, device=self.device)
        self.momentum = 0.9

        self.observation_spec = Composite({
            "observation": Unbounded(shape=(self.n_agents, 29), device=self.device),
        })

        low = torch.tensor([0.0 -math.pi], device=self.device).repeat(self.n_agents, 1)
        high = torch.tensor([self.max_speed, math.pi], device=self.device).repeat(self.n_agents, 1)
        self.action_spec = Composite({
            "action": Bounded(
                low=low,
                high=high,
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
        angles = 2 * math.pi * torch.rand(self.n_agents, 1, device=self.device)
        radii = self.center_radius * torch.sqrt(torch.rand(self.n_agents, 1, device=self.device))
        self.agent_positions = torch.cat(
            [radii * torch.cos(angles), radii * torch.sin(angles)], dim=1
        ).to(self.device)

    def _init_rooms(self) -> None:
        """Initializes rooms evenly spaced around the arena circumference, pulled in from the edge."""
        edge_margin = 0.2 * self.room_radius
        angles = torch.linspace(0.0, 2 * math.pi, steps=self.n_rooms + 1, device=self.device)[:-1]

        radius = self.arena_radius - self.room_radius - edge_margin
        self.room_positions = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1).to(self.device)

        self.room_occupancy = torch.zeros(self.n_rooms, dtype=torch.long, device=self.device)

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
        pos = self.agent_positions
        vel = self.agent_velocity
        facing = self.agent_facing % (2 * math.pi)  # Normalize

        n_agents = self.n_agents
        device = self.device

        # === Center Information ===
        vec_to_center = -pos
        dist_to_center = vec_to_center.norm(dim=1, keepdim=True)
        angle_to_center = torch.atan2(vec_to_center[:, 1], vec_to_center[:, 0])
        rel_angle_to_center = ((angle_to_center - facing + math.pi) % (2 * math.pi)) - math.pi
        rel_center_cos = torch.cos(rel_angle_to_center).unsqueeze(1)
        rel_center_sin = torch.sin(rel_angle_to_center).unsqueeze(1)
        dist_to_center_edge = dist_to_center - self.center_radius

        # === Closest Agent Information ===
        dist_matrix = torch.cdist(pos, pos, p=2)
        dist_matrix.fill_diagonal_(float('inf'))
        closest_agent_dists, closest_agent_indices = dist_matrix.min(dim=1)
        vec_to_closest_agent = self.agent_positions[closest_agent_indices] - pos
        angle_to_agent = torch.atan2(vec_to_closest_agent[:, 1], vec_to_closest_agent[:, 0])
        rel_angle_to_agent = ((angle_to_agent - facing + math.pi) % (2 * math.pi)) - math.pi
        rel_agent_cos = torch.cos(rel_angle_to_agent).unsqueeze(1)
        rel_agent_sin = torch.sin(rel_angle_to_agent).unsqueeze(1)

        # === Closest Room Information ===
        dist_to_rooms = torch.cdist(pos, self.room_positions, p=2)
        closest_room_dists, closest_room_indices = dist_to_rooms.min(dim=1)
        closest_room_pos = self.room_positions[closest_room_indices]
        vec_to_closest_room = closest_room_pos - pos
        angle_to_room = torch.atan2(vec_to_closest_room[:, 1], vec_to_closest_room[:, 0])
        rel_angle_to_room = ((angle_to_room - facing + math.pi) % (2 * math.pi)) - math.pi
        rel_room_cos = torch.cos(rel_angle_to_room).unsqueeze(1)
        rel_room_sin = torch.sin(rel_angle_to_room).unsqueeze(1)
        dist_to_closest_room_edge = closest_room_dists - self.room_radius

        # === Room Occupancy ===
        room_occ = self.room_occupancy.float().unsqueeze(1)
        closest_room_occ = room_occ[closest_room_indices]

        inside_room_mask = dist_to_rooms <= self.room_radius
        num_agents_in_rooms = inside_room_mask.sum(dim=0)
        num_agents_in_closest_room = num_agents_in_rooms[closest_room_indices].unsqueeze(1).float()

        # In-room boolean flag
        in_room = (closest_room_dists <= self.room_radius).unsqueeze(1).float()

        # One-hot room index
        assigned_room_oh = torch.nn.functional.one_hot(closest_room_indices, num_classes=self.n_rooms).float()

        # === Velocity Info ===
        speed = vel.norm(dim=1, keepdim=True)
        velocity_angle = torch.atan2(vel[:, 1], vel[:, 0])
        rel_velocity_angle = ((velocity_angle - facing + math.pi) % (2 * math.pi)) - math.pi
        rel_velocity_cos = torch.cos(rel_velocity_angle).unsqueeze(1)
        rel_velocity_sin = torch.sin(rel_velocity_angle).unsqueeze(1)

        # Add vx, vy explicitly
        vx_vy = vel  # shape (n_agents, 2)

        # === Facing Info ===
        facing_cos = torch.cos(facing).unsqueeze(1)
        facing_sin = torch.sin(facing).unsqueeze(1)

        # === Phase Info ===
        phase_tensor = torch.zeros((n_agents, 2), device=device)
        if self.phase == "claiming":
            phase_tensor[:, 1] = 1.0
        else:
            phase_tensor[:, 0] = 1.0

        capacity_tensor = torch.full((n_agents, 1), self.room_capacity, device=device)

        # === Final Observation Vector ===
        obs = torch.cat([
            dist_to_center,                          # (n_agents, 1)
            dist_to_center_edge,                     # (n_agents, 1)
            rel_center_cos,                          # (n_agents, 1)
            rel_center_sin,                          # (n_agents, 1)

            closest_agent_dists.unsqueeze(1),        # (n_agents, 1)
            rel_agent_cos,                           # (n_agents, 1)
            rel_agent_sin,                           # (n_agents, 1)

            closest_room_dists.unsqueeze(1),         # (n_agents, 1)
            dist_to_closest_room_edge.unsqueeze(1),  # (n_agents, 1)
            closest_room_occ,                        # (n_agents, 1)
            num_agents_in_closest_room,              # (n_agents, 1)
            in_room,                                 # (n_agents, 1)

            rel_room_cos,                            # (n_agents, 1)
            rel_room_sin,                            # (n_agents, 1)

            speed,                                   # (n_agents, 1)
            rel_velocity_cos,                        # (n_agents, 1)
            rel_velocity_sin,                        # (n_agents, 1)
            vx_vy,                                   # (n_agents, 2)

            capacity_tensor,                         # (n_agents, 1)
            phase_tensor,                            # (n_agents, 2)
            facing_cos, facing_sin,                  # (n_agents, 2)

            assigned_room_oh                         # (n_agents, n_rooms)
        ], dim=1)

        # Safety: clean up any NaN/infs
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            obs = torch.nan_to_num(obs)

        return obs

    def _compute_rewards(self) -> Tensor:
        """Aggregates rewards from reward modules."""
        rewards = torch.zeros((self.n_agents, 1), device=self.device)
        if self.reward_managers and self.phase in self.reward_managers:
            rewards = self.reward_managers[self.phase](self)
        else:
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

        self.agent_facing = 2 * math.pi * torch.rand(self.n_agents, device=self.device)
        self.agent_velocity = torch.zeros(self.n_agents, 2, device=self.device)

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

        # Clamp thrust to [0, 1] as per action spec, then scale by max_speed
        thrust = actions[:, 0].clamp(0.0, 1.0) * self.max_speed
        turn = actions[:, 1].clamp(-math.pi, math.pi)

        # Update facing direction
        self.agent_facing = (self.agent_facing + turn) % (2 * math.pi)

        # Compute thrust vector in facing direction
        direction_vectors = torch.stack([
            torch.cos(self.agent_facing),
            torch.sin(self.agent_facing)
        ], dim=1)

        # Apply momentum: blend old velocity and new thrust
        new_velocity = direction_vectors * thrust.unsqueeze(1)
        self.agent_velocity = self.momentum * self.agent_velocity + (1 - self.momentum) * new_velocity

        # Clamp velocity magnitude to max_speed
        speed = self.agent_velocity.norm(dim=1, keepdim=True)
        max_speed_tensor = torch.full_like(speed, self.max_speed)
        clipped_speed = torch.min(speed, max_speed_tensor)
        self.agent_velocity = self.agent_velocity / (speed + 1e-8) * clipped_speed

        # Update position
        self.agent_positions += self.agent_velocity

        self._enforce_boundaries()

        # Get room occupancy information
        room_dists = torch.cdist(self.agent_positions, self.room_positions)
        assignments = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)

        # First determine which agents are in which rooms
        for i in range(self.n_agents):
            close_rooms = (room_dists[i] < self.room_radius).nonzero(as_tuple=True)[0]
            if close_rooms.numel() > 0:
                closest = close_rooms[room_dists[i, close_rooms].argmin()]
                assignments[i] = closest

        # Calculate room occupancies
        self.room_occupancy = torch.bincount(assignments[assignments >= 0], minlength=self.n_rooms)

        # Phase transition
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
