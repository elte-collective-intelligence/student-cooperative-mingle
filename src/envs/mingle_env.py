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
        self.max_speed = 0.3  # Moderate speed for exploration
        self.room_radius = room_radius
        self.room_capacity = room_capacity
        self.reward_modules = reward_modules or []
        self.reward_managers = reward_managers
        self.phase_mode = phase_mode

        self.agent_positions = torch.zeros(self.n_agents, 2, device=self.device)
        self.room_positions = None
        self.room_occupancy = None

        # Track room entry times for each agent (-1 = not in any room)
        self.room_entry_time = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.agent_current_room = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.forced_to_leave = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        # ============ COMMUNICATION SYSTEM ============
        # Leader-follower pairing with "follow me" and "full" messages
        # - Leaders (n/2 agents) broadcast "follow me"
        # - Followers choose a leader to follow
        # - If leader already has a follower, broadcasts "full"
        self.enable_communication = False  # Set to True to enable
        self.is_leader = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.leader_message = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)  # 0=follow_me, 1=full
        self.following = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)  # Who each agent follows
        self.follower_count = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)  # How many followers each leader has

        # Observation includes communication: base(14) + comm(4) = 18
        obs_dim = 18 if self.enable_communication else 14
        self.observation_spec = Composite({
            "observation": Unbounded(shape=(self.n_agents, obs_dim), device=self.device),
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
        edge_margin = 0.2 * self.room_radius  # ORIGINAL value restored
        angles = torch.linspace(0.0, 2 * math.pi, steps=self.n_rooms + 1)[:-1]

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

    def _enforce_boundaries(self, actions: Tensor = None) -> None:
        """
        Prevents agents from moving outside the arena.
        When hitting border, agents bounce back (reflect velocity).
        """
        dists = self.agent_positions.norm(dim=1)
        outside = dists > self.arena_radius
        if outside.any():
            # Get unit vector pointing toward center
            center_dir = -self.agent_positions[outside] / dists[outside].unsqueeze(1)

            # Push agents back inside
            self.agent_positions[outside] = center_dir * (-self.arena_radius * 0.95)

            # If we have velocities, reflect them (bounce)
            if hasattr(self, 'last_velocity') and self.last_velocity is not None:
                # Reflect velocity: v' = v - 2(v·n)n where n is normal (toward center)
                v = self.last_velocity[outside]
                dot = (v * center_dir).sum(dim=1, keepdim=True)
                self.last_velocity[outside] = v - 2 * dot * center_dir

    def _compute_observations(self) -> Tensor:
        """
        Constructs a 14-dimensional observation vector for each agent.

        PHASE GATING: During spinning phase, room information is HIDDEN.
        Agents cannot see room positions/occupancy until claiming phase begins.
        This matches the assignment spec where rooms are "revealed" after spinning.
        """
        pos = self.agent_positions
        norm = pos.norm(dim=1, keepdim=True)

        dist_to_center = norm / self.center_radius
        direction_to_center = -pos / (norm + 1e-8)
        dist_to_center_edge = (norm - self.center_radius) / self.center_radius

        # PHASE GATING: Room info only available during claiming
        if self.phase == "claiming":
            # Distance to rooms (REVEALED)
            room_dists = torch.cdist(pos, self.room_positions)
            closest_room = room_dists.argmin(dim=1)

            # ============ TRACK ROOM ENTRY TIMES ============
            # Check which agents are currently inside rooms
            assignments = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
            for i in range(self.n_agents):
                close_rooms = (room_dists[i] < self.room_radius).nonzero(as_tuple=True)[0]
                if close_rooms.numel() > 0:
                    closest = close_rooms[room_dists[i, close_rooms].argmin()]
                    assignments[i] = closest

            # Update entry times
            for i in range(self.n_agents):
                current_room = assignments[i].item()
                prev_room = self.agent_current_room[i].item()

                if current_room >= 0 and prev_room != current_room:
                    # Agent just entered a new room - record entry time
                    self.room_entry_time[i] = self.current_step
                    self.agent_current_room[i] = current_room
                elif current_room < 0:
                    # Agent left all rooms
                    self.room_entry_time[i] = -1
                    self.agent_current_room[i] = -1

            self.room_occupancy = torch.bincount(assignments[assignments >= 0], minlength=self.n_rooms)

            # ============ FIND OVERFLOW AGENTS (LAST TO ENTER) ============
            self.forced_to_leave.fill_(False)

            for room_idx in range(self.n_rooms):
                if self.room_occupancy[room_idx] > self.room_capacity:
                    # Find all agents in this room
                    agents_in_room = (assignments == room_idx).nonzero(as_tuple=True)[0]

                    if agents_in_room.numel() > self.room_capacity:
                        # Get entry times for these agents
                        entry_times = self.room_entry_time[agents_in_room]

                        # Sort by entry time (ascending) - earliest first
                        sorted_indices = entry_times.argsort()
                        sorted_agents = agents_in_room[sorted_indices]

                        # Agents who entered LAST must leave (overflow agents)
                        overflow_agents = sorted_agents[self.room_capacity:]
                        self.forced_to_leave[overflow_agents] = True

            # ============ COMPUTE ROOM DIRECTIONS ============
            # ALL agents should be directed to nearest NON-FULL room
            # Never direct an agent to a room that's already at capacity!

            nearest_room_dir = torch.zeros((self.n_agents, 2), device=self.device)
            dist_to_room = torch.zeros((self.n_agents, 1), device=self.device)

            for i in range(self.n_agents):
                current_room = self.agent_current_room[i].item()

                # Check if agent is already a valid occupant in a room
                is_valid_in_room = (current_room >= 0 and
                                   not self.forced_to_leave[i] and
                                   self.room_occupancy[current_room] <= self.room_capacity)

                if is_valid_in_room:
                    # Agent is validly in a room - point to current room (stay)
                    delta = self.room_positions[current_room] - pos[i]
                    nearest_room_dir[i] = delta / (delta.norm() + 1e-8)
                    dist_to_room[i] = room_dists[i, current_room].item() / self.arena_radius
                else:
                    # Agent needs to find a room - find nearest NON-FULL room
                    best_room = -1
                    best_dist = float('inf')

                    for r in range(self.n_rooms):
                        # Skip current room if agent is forced to leave
                        if self.forced_to_leave[i] and r == current_room:
                            continue

                        # Only consider rooms with available capacity
                        if self.room_occupancy[r] < self.room_capacity:
                            d = room_dists[i, r].item()
                            if d < best_dist:
                                best_dist = d
                                best_room = r

                    if best_room >= 0:
                        # Point to nearest available room
                        delta = self.room_positions[best_room] - pos[i]
                        nearest_room_dir[i] = delta / (delta.norm() + 1e-8)
                        dist_to_room[i] = best_dist / self.arena_radius
                    else:
                        # All rooms full - point to center (wait area)
                        delta = -pos[i]  # Direction to center
                        norm = delta.norm()
                        if norm > 0.1:
                            nearest_room_dir[i] = delta / norm
                        else:
                            nearest_room_dir[i] = torch.zeros(2, device=self.device)
                        dist_to_room[i] = 1.0

            # Signed distance to edge of the closest room (normalized)
            raw_room_dist = room_dists[torch.arange(self.n_agents), closest_room].unsqueeze(1)
            signed_dist_to_room_edge = (self.room_radius - raw_room_dist) / self.room_radius

            capacity_tensor = torch.full((self.n_agents, 1), self.room_capacity, device=self.device)
            occupancy_tensor = self.room_occupancy[closest_room].unsqueeze(1).float() / self.room_capacity
        else:
            # SPINNING PHASE: Room info is HIDDEN (all zeros/neutral values)
            # Agents cannot pre-position because they don't know where rooms are
            dist_to_room = torch.ones((self.n_agents, 1), device=self.device)  # Max distance
            nearest_room_dir = torch.zeros((self.n_agents, 2), device=self.device)  # No direction
            signed_dist_to_room_edge = torch.zeros((self.n_agents, 1), device=self.device)  # Neutral
            capacity_tensor = torch.zeros((self.n_agents, 1), device=self.device)  # Hidden
            occupancy_tensor = torch.zeros((self.n_agents, 1), device=self.device)  # Hidden
            # Don't update room_occupancy during spinning
            self.room_occupancy = torch.zeros(self.n_rooms, dtype=torch.long, device=self.device)

        # Distance to nearest agent (always visible - for collision avoidance)
        agent_dists = torch.cdist(pos, pos)
        agent_dists.fill_diagonal_(float("inf"))
        nearest_agent_idx = agent_dists.argmin(dim=1)
        nearest_directions = torch.stack([
            (pos[j] - pos[i]) / ((pos[j] - pos[i]).norm() + 1e-8)
            for i, j in enumerate(nearest_agent_idx)
        ])
        dist_to_agent = agent_dists.min(dim=1, keepdim=True).values / self.arena_radius

        phase_flag = torch.full((self.n_agents, 1), 1.0 if self.phase == "claiming" else 0.0, device=self.device)

        # ============ COMMUNICATION UPDATE ============
        if self.enable_communication:
            # Update follower-leader pairing based on "full" messages
            followers = (~self.is_leader).nonzero(as_tuple=True)[0]
            leaders = self.is_leader.nonzero(as_tuple=True)[0]

            for f in followers:
                current_leader = self.following[f].item()
                if current_leader >= 0:
                    # Check if current leader says "full" AND has more than 1 follower
                    if self.leader_message[current_leader] == 1 and self.follower_count[current_leader] > 1:
                        # Find another leader who is NOT full
                        available_leaders = leaders[self.leader_message[leaders] == 0]
                        if len(available_leaders) > 0:
                            # Switch to available leader
                            self.follower_count[current_leader] -= 1
                            new_leader = available_leaders[torch.randint(len(available_leaders), (1,)).item()]
                            self.following[f] = new_leader
                            self.follower_count[new_leader] += 1

            # Update leader messages based on follower count
            self.leader_message.fill_(0)  # Reset to "follow me"
            self.leader_message[self.follower_count > 1] = 1  # "full" if more than 1 follower

            # Compute communication features for observation
            # comm_obs: [is_leader, leader_message, direction_to_leader_x, direction_to_leader_y]
            comm_obs = torch.zeros((self.n_agents, 4), device=self.device)

            for i in range(self.n_agents):
                comm_obs[i, 0] = 1.0 if self.is_leader[i] else 0.0
                if self.is_leader[i]:
                    comm_obs[i, 1] = float(self.leader_message[i])  # 0=follow_me, 1=full
                else:
                    # Follower: direction to their leader
                    leader_idx = self.following[i].item()
                    if leader_idx >= 0:
                        delta = pos[leader_idx] - pos[i]
                        norm = delta.norm()
                        if norm > 0.01:
                            comm_obs[i, 2:4] = delta / norm
                        # Also include leader's message
                        comm_obs[i, 1] = float(self.leader_message[leader_idx])

        # Build observation
        base_obs = torch.cat([
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

        if self.enable_communication:
            obs = torch.cat([base_obs, comm_obs], dim=1)
        else:
            obs = base_obs

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

        # Reset reward history for fairness metrics (GINI coefficient calculation)
        self.last_rewards = torch.zeros((self.n_agents, 1), device=self.device)

        # Initialize boundary violation and action tracking
        self.boundary_violation = torch.zeros(self.n_agents, device=self.device)
        self.last_action_magnitudes = torch.zeros(self.n_agents, device=self.device)

        # Track previous target room for switch penalty
        self.prev_target_room = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)

        # Reset room entry tracking
        self.room_entry_time = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.agent_current_room = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.forced_to_leave = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        # ============ RESET COMMUNICATION ============
        if self.enable_communication:
            # Randomly designate n/2 agents as leaders
            n_leaders = self.n_agents // 2
            leader_indices = torch.randperm(self.n_agents)[:n_leaders]
            self.is_leader = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
            self.is_leader[leader_indices] = True

            # All leaders start with "follow me" message (0)
            self.leader_message = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

            # Reset follower assignments
            self.following = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
            self.follower_count = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

            # Followers randomly pick initial leaders
            followers = (~self.is_leader).nonzero(as_tuple=True)[0]
            leaders = self.is_leader.nonzero(as_tuple=True)[0]

            for f in followers:
                # Pick a random leader
                chosen = leaders[torch.randint(len(leaders), (1,)).item()]
                self.following[f] = chosen
                self.follower_count[chosen] += 1

            # Leaders with more than 1 follower say "full"
            self.leader_message[self.follower_count > 1] = 1  # 1 = "full"

        # Track success state for early termination
        self.success_achieved = False

        # Track velocity for border bounce
        self.last_velocity = torch.zeros((self.n_agents, 2), device=self.device)

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

        # Store velocity for bounce calculations
        self.last_velocity = actions.clone()

        self.agent_positions += actions

        # COMPUTE BOUNDARY VIOLATION BEFORE CLAMPING
        # This captures the "attempted exit" behavior for penalty computation
        dists_before_clamp = self.agent_positions.norm(dim=1)
        self.boundary_violation = torch.clamp(dists_before_clamp - self.arena_radius, min=0.0)

        # Store action magnitudes for regularization
        self.last_action_magnitudes = actions.norm(dim=1)

        # Enforce boundaries with bounce
        self._enforce_boundaries(actions)

        if self.phase == "spinning" and self.phase_mode == "both" and self.current_step >= self.spinning_duration:
            self.phase = "claiming"

        if self.phase == "spinning":
            in_center = self.agent_positions.norm(dim=1) <= self.center_radius
            if in_center.any():
                self._rotate_positions(0.05)

        obs = self._compute_observations()
        reward = self._compute_rewards()

        # Save last reward for fairness metrics (GINI coefficient calculation)
        self.last_rewards = reward.detach().clone()

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