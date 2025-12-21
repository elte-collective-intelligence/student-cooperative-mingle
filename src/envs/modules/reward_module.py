import torch
from torch import Tensor
from typing import Tuple

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
        # FIX: Handle "both" phase_mode correctly
        # "both" means active in ALL phases, not that phase must equal "both"
        if not self.active:
            return torch.zeros((env.n_agents, 1), device=env.device)

        # Check phase compatibility
        if self.phase_mode == "both":
            # Active in all phases
            pass
        elif env.phase != self.phase_mode:
            # Only active in specific phase
            return torch.zeros((env.n_agents, 1), device=env.device)

        return self._reward(env)

    def _activate(self):
        self.active = True

    def _deactivate(self):
        self.active = False

    def _reward(self, env: "MingleEnv") -> Tensor:
        raise NotImplementedError("Subclasses must implement the _reward method.")


class CloseToCenterReward(RewardModule):
    """Continuous reward based on distance to center."""
    def __init__(self, phase_mode: str = "both"):
        super().__init__(phase_mode=phase_mode)

    def _reward(self, env: "MingleEnv") -> Tensor:
        dists = env.agent_positions.norm(dim=1, keepdim=True)
        return 1.0 - dists.clamp(0, env.center_radius) / env.center_radius


class InsideCenterReward(RewardModule):
    """
    Reward for being inside center circle (spinning phase).
    Dense version: small positive per step inside, penalty per step outside.
    """
    def __init__(self, phase_mode: str = "both", inside_reward: float = 1.0, outside_penalty: float = 1.0):
        super().__init__(phase_mode=phase_mode)
        self.inside_reward = inside_reward
        self.outside_penalty = outside_penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        dists = env.agent_positions.norm(dim=1, keepdim=True) / env.center_radius
        inside_mask = (dists <= 1.0).float()
        outside_mask = 1.0 - inside_mask
        return (inside_mask * self.inside_reward) - (outside_mask * self.outside_penalty)


class CollisionAvoidanceReward(RewardModule):
    """Penalty for being too close to other agents."""
    def __init__(self, phase_mode: str = "both", min_distance: float = 0.5, penalty: float = 1.0):
        super().__init__(phase_mode=phase_mode)
        self.min_distance = min_distance
        self.penalty = penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        dists = torch.cdist(env.agent_positions, env.agent_positions)
        dists.fill_diagonal_(self.min_distance + 1.0)
        too_close = (dists < self.min_distance).any(dim=1, keepdim=True).float()
        return -self.penalty * too_close


class WallPenaltyReward(RewardModule):
    """
    Strong penalty for going near arena borders/walls.
    This prevents agents from running to the edges of the arena.
    """
    def __init__(self, phase_mode: str = "both", wall_distance: float = 1.5, penalty: float = 5.0):
        super().__init__(phase_mode=phase_mode)
        self.wall_distance = wall_distance
        self.penalty = penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        # Distance from center
        dists_from_center = env.agent_positions.norm(dim=1, keepdim=True)
        # Distance to wall
        dists_to_wall = env.arena_radius - dists_from_center
        # Penalty zone: within wall_distance of the border
        in_penalty_zone = (dists_to_wall < self.wall_distance).float()
        # Stronger penalty the closer to wall
        proximity = (1.0 - dists_to_wall / self.wall_distance).clamp(0, 1)
        return -self.penalty * in_penalty_zone * proximity


class GetToRoomReward(RewardModule):
    """
    Reward for moving towards rooms.

    FIXED: Now provides LINEAR gradient toward rooms instead of flat reward.
    Agents get higher reward the closer they are to any room.
    """
    def __init__(
        self,
        phase_mode: str = "both",
        max_reward: float = 1.0,
        full_room_penalty: float = 1.0,
    ):
        super().__init__(phase_mode=phase_mode)
        self.max_reward = max_reward
        self.full_room_penalty = full_room_penalty

    def _reward(self, env: "MingleEnv") -> Tensor:
        room_dists = torch.cdist(env.agent_positions, env.room_positions)
        min_dists, closest_room_indices = room_dists.min(dim=1, keepdim=True)

        closest_room_occupancy = env.room_occupancy[closest_room_indices.squeeze(1)]
        room_capacity = env.room_capacity
        is_full = (closest_room_occupancy >= room_capacity).float().unsqueeze(1)

        # LINEAR GRADIENT: reward = max_reward * (1 - distance/max_distance)
        # At distance 0: reward = max_reward
        # At distance = arena_radius: reward = 0
        # This gives agents a clear signal about which direction leads to rooms!
        max_possible_dist = env.arena_radius + env.room_radius
        normalized_dist = (min_dists / max_possible_dist).clamp(0, 1)
        proximity_reward = self.max_reward * (1.0 - normalized_dist)

        # Apply only to non-full rooms
        reward = proximity_reward * (1 - is_full)
        penalty = self.full_room_penalty * normalized_dist * is_full  # Penalize being far from full rooms less

        return reward - penalty


class StayInRoomReward(RewardModule):
    """
    Dense reward for staying in rooms (claiming phase).

    KEY FEATURES:
    - High reward for being in a room with valid capacity
    - TIME-BASED PENALTY: Penalty for being outside increases over time
    - LAST-AGENT PENALTY: Only penalize agents who entered LAST when room is overfilled
      (agents closest to room center are "winners", furthest are penalized)
    """
    def __init__(
        self,
        phase_mode: str = "both",
        max_reward: float = 1.0,
        outside_penalty: float = 1.0,
        overfill_penalty: float = 2.0,
        time_penalty_start: int = 50,  # Start increasing penalty after this step
        time_penalty_scale: float = 0.02,  # How fast penalty increases
    ):
        super().__init__(phase_mode=phase_mode)
        self.max_reward = max_reward
        self.outside_penalty = outside_penalty
        self.overfill_penalty = overfill_penalty
        self.time_penalty_start = time_penalty_start
        self.time_penalty_scale = time_penalty_scale

    def _reward(self, env: "MingleEnv") -> Tensor:
        n_agents = env.n_agents
        device = env.device
        rewards = torch.zeros((n_agents, 1), device=device)

        room_dists = torch.cdist(env.agent_positions, env.room_positions)
        room_capacity = env.room_capacity

        # For each room, find which agents are inside
        for room_idx in range(env.n_rooms):
            room_dist = room_dists[:, room_idx]
            inside_mask = room_dist < env.room_radius
            inside_indices = inside_mask.nonzero(as_tuple=True)[0]

            if inside_indices.numel() == 0:
                continue

            # Sort agents by distance to room center (closest first = winners)
            distances = room_dist[inside_indices]
            sorted_order = distances.argsort()
            sorted_agents = inside_indices[sorted_order]

            # First room_capacity agents are valid, rest are penalized
            valid_agents = sorted_agents[:room_capacity]
            excess_agents = sorted_agents[room_capacity:]

            # Valid agents get reward
            rewards[valid_agents] += self.max_reward

            # ONLY excess agents (entered LAST / furthest from center) get penalized
            if excess_agents.numel() > 0:
                rewards[excess_agents] -= self.overfill_penalty

        # Agents outside all rooms
        in_any_room = (room_dists < env.room_radius).any(dim=1)
        outside_agents = ~in_any_room

        # TIME-BASED PENALTY: increases over time
        current_step = env.current_step if hasattr(env, 'current_step') else 0
        time_factor = max(0, current_step - self.time_penalty_start) * self.time_penalty_scale
        dynamic_penalty = self.outside_penalty * (1.0 + time_factor)

        rewards[outside_agents] -= dynamic_penalty

        return rewards


class RoomSpreadReward(RewardModule):
    """
    Reward for spreading agents across different rooms (coordination reward).

    This encourages agents to distribute evenly rather than all going to same room.
    """
    def __init__(self, phase_mode: str = "claiming", spread_bonus: float = 0.5):
        super().__init__(phase_mode=phase_mode)
        self.spread_bonus = spread_bonus

    def _reward(self, env: "MingleEnv") -> Tensor:
        n_agents = env.n_agents
        rewards = torch.zeros((n_agents, 1), device=env.device)

        # Count rooms with exactly correct number of agents (2)
        rooms_at_capacity = (env.room_occupancy == env.room_capacity).sum().float()
        # Normalize by number of rooms
        spread_ratio = rooms_at_capacity / env.n_rooms

        # All agents get a bonus when rooms are well-distributed
        rewards += self.spread_bonus * spread_ratio

        return rewards


class LeaveToRoomReward(RewardModule):
    """
    Reward for leaving center and heading towards rooms during claiming phase.

    This provides a dense signal to encourage agents to exit center zone.
    """
    def __init__(self, phase_mode: str = "claiming", leave_reward: float = 0.5):
        super().__init__(phase_mode=phase_mode)
        self.leave_reward = leave_reward

    def _reward(self, env: "MingleEnv") -> Tensor:
        # Distance from center
        center_dists = env.agent_positions.norm(dim=1, keepdim=True)

        # Distance to closest room
        room_dists = torch.cdist(env.agent_positions, env.room_positions)
        min_room_dists = room_dists.min(dim=1, keepdim=True)[0]

        # Reward for being far from center AND close to a room
        # Normalized so max is ~1
        center_bonus = (center_dists / env.center_radius).clamp(0, 2) / 2
        room_bonus = 1.0 - (min_room_dists / env.arena_radius).clamp(0, 1)

        return self.leave_reward * center_bonus * room_bonus


class UnifiedCooperativeReward(RewardModule):
    """
    Unified reward function with STRICT PHASE SEPARATION per assignment spec.

    KEY FIXES (v4 - Final):
    1. Uses env's AUTHORITATIVE assignments (agent_in_room, excluded, room_occupancy)
    2. Feasibility uses winner status, not "physically inside"
    3. Excluded agents masked from room they just lost
    4. Hysteresis on target selection (margin-based)
    5. POTENTIAL-BASED progress shaping: reward = α(Φ(s')-Φ(s)), not absolute
    6. Split room reward: enter_bonus (one-time) + hold_bonus (per-step)
    7. Soft crowding term to reduce stampedes
    8. Distance-to-EDGE normalization with proper max

    SPINNING PHASE (rooms hidden):
    - Reward staying in center circle
    - Penalize leaving center
    - Near-wall penalty active

    CLAIMING PHASE (rooms revealed):
    - Enter bonus when becoming a winner (one-time)
    - Hold bonus while staying a winner (per-step, small)
    - Potential-based shaping for non-winners
    - Excluded agents redirected to OTHER feasible rooms
    """
    def __init__(
        self,
        phase_mode: str = "both",
        reward_scale: float = 1.0,
        # Spinning phase
        center_stay_bonus: float = 0.5,
        center_leave_penalty: float = 0.3,
        # Claiming phase - NEW REWARD STRUCTURE
        enter_nonfull_bonus: float = 5.0,    # HIGH reward for entering non-full room
        stay_in_room_bonus: float = 0.2,     # Small bonus per step staying in valid room
        leave_nonfull_penalty: float = 3.0,  # HIGH penalty for leaving a non-full room
        in_full_room_penalty: float = 4.0,   # HIGH penalty for being in overcrowded room
        leave_full_room_penalty: float = 0.5, # SMALL penalty for leaving full room (time pressure)
        center_penalty: float = 0.5,
        excluded_penalty: float = 0.3,       # For excluded agents (3rd+ in room)
        ignored_warning_penalty: float = 2.0,  # Penalty for ignoring "Room full" warning
        # Boundary control - BALANCED to allow room access while preventing border-seeking
        wall_margin: float = 1.5,            # Penalty zone starts at 6.5 (rooms at 5.0 are safe)
        near_wall_penalty: float = 3.0,      # Strong but not excessive
        boundary_penalty: float = 15.0,      # Strong violation penalty
        action_regularization: float = 0.05,
        # Collision
        collision_penalty: float = 0.3,
        # Progress shaping
        progress_scale: float = 0.5,         # Scale for Φ(s')-Φ(s)
        crowding_cost: float = 0.3,          # Soft cost for near-full rooms
        # Fairness
        gini_penalty: float = 0.2,
        # Success
        success_bonus: float = 3.0,
        # Hysteresis
        target_hysteresis: float = 0.3,      # Margin to prevent target flipping
    ):
        super().__init__(phase_mode=phase_mode)
        self.reward_scale = reward_scale
        self.center_stay_bonus = center_stay_bonus
        self.center_leave_penalty = center_leave_penalty
        self.enter_nonfull_bonus = enter_nonfull_bonus
        self.stay_in_room_bonus = stay_in_room_bonus
        self.leave_nonfull_penalty = leave_nonfull_penalty
        self.in_full_room_penalty = in_full_room_penalty
        self.leave_full_room_penalty = leave_full_room_penalty
        self.center_penalty = center_penalty
        self.excluded_penalty = excluded_penalty
        self.ignored_warning_penalty = ignored_warning_penalty

        # Track previous room state for detecting entries/exits
        self.prev_in_room = None
        self.prev_room_idx = None
        self.prev_room_was_full = None
        self.wall_margin = wall_margin
        self.near_wall_penalty = near_wall_penalty
        self.boundary_penalty = boundary_penalty
        self.action_regularization = action_regularization
        self.collision_penalty = collision_penalty
        self.progress_scale = progress_scale
        self.crowding_cost = crowding_cost
        self.gini_penalty = gini_penalty
        self.success_bonus = success_bonus
        self.target_hysteresis = target_hysteresis

        # State tracking (persistent across steps)
        self.prev_target = None
        self.prev_potential = None
        self.prev_winners = None

        # Cooldown to prevent excluded agents from immediately retargeting same room
        self.cooldown_steps = 12       # tune: 8-20
        self.cooldown_timer = None     # (n_agents,) long, steps remaining
        self.cooldown_room = None      # (n_agents,) long, room to avoid while timer > 0

    def _get_feasible_target_with_hysteresis(
        self, env, room_dists: Tensor, winners: Tensor, winner_room: Tensor,
        excluded: Tensor, excluded_room_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute nearest FEASIBLE room for each agent with hysteresis + cooldown.

        A room is feasible if:
        - occupancy < capacity, OR
        - agent is already a WINNER in that room (can stay)

        Cooldown: Excluded agents are blocked from their lost room for cooldown_steps.

        Hysteresis: Only switch target if new room edge is significantly closer.

        Returns:
            feasible_target: (n_agents,) index of target room
            feasible_dist: (n_agents,) distance to room EDGE (0 if inside)
        """
        n_agents = room_dists.shape[0]
        n_rooms = room_dists.shape[1]
        device = room_dists.device
        room_radius = env.room_radius
        room_capacity = env.room_capacity
        room_occupancy = env.room_occupancy if hasattr(env, 'room_occupancy') else torch.zeros(n_rooms, dtype=torch.long, device=device)

        # Distance to room EDGE (0 if already inside)
        edge_dists = (room_dists - room_radius).clamp(min=0.0)  # [A, R]

        # Build feasibility mask: [n_agents, n_rooms]
        # Start with rooms that have space (using authoritative winners-only occupancy)
        free_rooms = (room_occupancy < room_capacity)  # [R]
        feasible = free_rooms.unsqueeze(0).expand(n_agents, n_rooms).clone()  # [A, R]

        # Winners can always keep their own room even if it appears full
        ar = torch.arange(n_agents, device=device)
        w_idx = winners.nonzero(as_tuple=True)[0]
        if w_idx.numel() > 0:
            feasible[w_idx, winner_room[w_idx]] = True

        # Apply cooldown mask: blocked rooms for agents with active cooldown
        if self.cooldown_timer is not None:
            cd_active = self.cooldown_timer > 0
            if cd_active.any():
                cd_idx = cd_active.nonzero(as_tuple=True)[0]
                for i in cd_idx:
                    cr = self.cooldown_room[i].item()
                    if cr >= 0:
                        feasible[i, cr] = False

        # Fallback: if no feasible rooms for an agent, allow all
        no_feasible = ~feasible.any(dim=1)
        if no_feasible.any():
            feasible[no_feasible] = True

        # Masked edge distances (inf for infeasible)
        masked = edge_dists.masked_fill(~feasible, float('inf'))
        best_dist, best_room = masked.min(dim=1)

        # Apply hysteresis: keep previous target unless new one is better by margin
        if self.prev_target is None:
            return best_room, best_dist

        prev = self.prev_target
        prev_feasible = feasible[ar, prev]
        prev_dist = edge_dists[ar, prev].masked_fill(~prev_feasible, float('inf'))

        # Stay with previous target unless new is significantly closer
        stay = prev_feasible & (prev_dist <= best_dist + self.target_hysteresis)
        target = torch.where(stay, prev, best_room)
        dist = torch.where(stay, prev_dist, best_dist)

        return target, dist

    def _compute_gini(self, room_occupancy: Tensor) -> float:
        """Compute Gini coefficient of room occupancy distribution."""
        occ = room_occupancy.float()
        n = len(occ)
        if occ.sum() == 0:
            return 0.0
        sorted_occ, _ = torch.sort(occ)
        cumsum = torch.cumsum(sorted_occ, dim=0)
        gini = 1.0 - 2.0 * cumsum.sum() / (n * occ.sum()) + 1.0 / n
        return max(0.0, min(1.0, gini.item()))

    def _reward(self, env: "MingleEnv") -> Tensor:
        """
        Compute rewards with STRICT phase separation and proper shaping.

        KEY FEATURES (v5):
        - Potential-based progress shaping: r = α(Φ(s')-Φ(s))
        - Split room rewards: enter_bonus (one-time) + hold_bonus (per-step)
        - Cooldown prevents excluded agents from bounce-back to same room
        - Switch penalty exempts forced switches (prev target became infeasible)
        - Dynamic max_edge_dist for stable normalization
        """
        n_agents = env.n_agents
        device = env.device
        arena_radius = env.arena_radius
        rewards = torch.zeros((n_agents, 1), device=device)

        positions = env.agent_positions
        center_radius = env.center_radius
        center_dists = positions.norm(dim=1)
        in_center = center_dists < center_radius

        # ============ PHASE-SPECIFIC REWARDS ============
        # NOTE: Complex boundary penalties REMOVED - they were causing issues
        # The environment already enforces boundaries via _enforce_boundaries()

        if env.phase == "spinning":
            # === SPINNING PHASE: Stay in center ===
            center_reward = torch.zeros(n_agents, device=device)
            center_reward[in_center] = self.center_stay_bonus
            center_reward[~in_center] = -self.center_leave_penalty

            # Gradient toward center for outside agents
            outside_agents = ~in_center
            if outside_agents.any():
                excess_dist = (center_dists[outside_agents] - center_radius) / center_radius
                center_reward[outside_agents] -= 0.3 * excess_dist

            rewards += center_reward.unsqueeze(1)

            # Reset claiming-phase state (including cooldown)
            self.prev_target = None
            self.prev_potential = None
            self.prev_winners = None
            self.prev_in_room = None
            self.prev_room_idx = None
            self.prev_room_was_full = None
            if self.cooldown_timer is not None:
                self.cooldown_timer.zero_()
                self.cooldown_room.fill_(-1)

        else:
            # === CLAIMING PHASE: Get to feasible rooms ===
            room_positions = env.room_positions
            room_radius = env.room_radius
            room_capacity = env.room_capacity
            n_rooms = env.n_rooms

            # Initialize cooldown state if needed
            if self.cooldown_timer is None or self.cooldown_timer.shape[0] != n_agents:
                self.cooldown_timer = torch.zeros(n_agents, dtype=torch.long, device=device)
                self.cooldown_room = torch.full((n_agents,), -1, dtype=torch.long, device=device)

            # Calculate room distances
            room_dists = torch.cdist(positions, room_positions)

            # ============ GET AUTHORITATIVE ASSIGNMENTS ============
            use_unified = hasattr(env, 'agent_in_room') and hasattr(env, 'agent_excluded')

            if use_unified:
                winners = env.agent_in_room
                winner_room = env.agent_room_idx if hasattr(env, 'agent_room_idx') else room_dists.argmin(dim=1)
                excluded = env.agent_excluded
                excluded_room_idx = env.excluded_room_idx if hasattr(env, 'excluded_room_idx') else torch.full((n_agents,), -1, dtype=torch.long, device=device)
            else:
                # Fallback: compute capacity-correct assignments locally
                winners = torch.zeros(n_agents, dtype=torch.bool, device=device)
                winner_room = torch.full((n_agents,), -1, dtype=torch.long, device=device)
                excluded = torch.zeros(n_agents, dtype=torch.bool, device=device)
                excluded_room_idx = torch.full((n_agents,), -1, dtype=torch.long, device=device)

                for r in range(n_rooms):
                    inside_mask = room_dists[:, r] < room_radius
                    inside_indices = inside_mask.nonzero(as_tuple=True)[0]
                    if inside_indices.numel() == 0:
                        continue
                    distances = room_dists[inside_indices, r]
                    sorted_order = distances.argsort()
                    sorted_agents = inside_indices[sorted_order]
                    winner_agents = sorted_agents[:room_capacity]
                    winners[winner_agents] = True
                    winner_room[winner_agents] = r
                    if sorted_agents.numel() > room_capacity:
                        loser_agents = sorted_agents[room_capacity:]
                        excluded[loser_agents] = True
                        excluded_room_idx[loser_agents] = r

            # ============ UPDATE COOLDOWN TIMERS ============
            # Decrement active cooldowns
            active_cd = self.cooldown_timer > 0
            self.cooldown_timer[active_cd] -= 1
            expired = self.cooldown_timer == 0
            self.cooldown_room[expired] = -1

            # Refresh cooldown for newly excluded agents
            ex = excluded & (excluded_room_idx >= 0)
            self.cooldown_timer[ex] = self.cooldown_steps
            self.cooldown_room[ex] = excluded_room_idx[ex]

            # ============ FEASIBLE TARGET WITH HYSTERESIS + COOLDOWN ============
            feasible_target, feasible_dist = self._get_feasible_target_with_hysteresis(
                env, room_dists, winners, winner_room, excluded, excluded_room_idx
            )

            # ============ COMPUTE POTENTIAL FOR SHAPING ============
            # feasible_dist is already edge distance (0 if inside room)
            # Dynamic max_edge_dist for stable normalization
            room_center_r = room_positions.norm(dim=1).max().item()
            max_edge_dist = arena_radius + room_center_r - room_radius

            potential = 1.0 - (feasible_dist / max_edge_dist).clamp(0, 1)  # [0, 1], 1 = at edge

            claiming_reward = torch.zeros(n_agents, device=device)

            # Get room occupancy
            if hasattr(env, 'room_occupancy'):
                occ = env.room_occupancy
            else:
                occ = torch.zeros(n_rooms, dtype=torch.long, device=device)

            # ============ NEW REWARD STRUCTURE ============
            # Track current room state for each agent
            current_in_room = winners.clone()  # agent is validly in a room
            current_room_idx = winner_room.clone()
            current_room_full = torch.zeros(n_agents, dtype=torch.bool, device=device)

            for i in range(n_agents):
                if current_in_room[i] and current_room_idx[i] >= 0:
                    room_occ = occ[current_room_idx[i]].item()
                    current_room_full[i] = room_occ >= room_capacity

            # Initialize previous state if needed
            if self.prev_in_room is None:
                self.prev_in_room = torch.zeros(n_agents, dtype=torch.bool, device=device)
                self.prev_room_idx = torch.full((n_agents,), -1, dtype=torch.long, device=device)
                self.prev_room_was_full = torch.zeros(n_agents, dtype=torch.bool, device=device)

            # ============ 1. ENTERED NON-FULL ROOM: HIGH REWARD ============
            # Agent was NOT in room before, now IS in a non-full room
            entered_room = current_in_room & ~self.prev_in_room
            entered_nonfull = entered_room & ~current_room_full
            claiming_reward[entered_nonfull] += self.enter_nonfull_bonus

            # ============ 2. STAYING IN VALID ROOM: SMALL BONUS ============
            # Agent was in room before and still in same room (not overcrowded)
            staying_in_room = current_in_room & self.prev_in_room & ~excluded
            claiming_reward[staying_in_room] += self.stay_in_room_bonus

            # ============ 3. LEFT NON-FULL ROOM: HIGH PENALTY ============
            # Agent WAS in a non-full room, now NOT in room (why leave a good spot?)
            left_room = self.prev_in_room & ~current_in_room
            left_nonfull = left_room & ~self.prev_room_was_full
            claiming_reward[left_nonfull] -= self.leave_nonfull_penalty

            # ============ 4. IN FULL ROOM (OVERCROWDED): HIGH PENALTY ============
            # Agent is excluded (3rd+ agent in room) - they shouldn't be there
            claiming_reward[excluded] -= self.in_full_room_penalty

            # ============ 5. LEFT FULL ROOM: SMALL PENALTY (time pressure) ============
            # Agent left a full room - still penalized but less (need to find new room)
            left_full = left_room & self.prev_room_was_full
            claiming_reward[left_full] -= self.leave_full_room_penalty

            # ============ 6. IGNORED "ROOM FULL" WARNING: HIGH PENALTY ============
            # Penalize agents who entered a room despite "Room full" warnings
            if hasattr(env, 'agent_ignored_warning'):
                for i in range(n_agents):
                    if env.agent_ignored_warning(i):
                        claiming_reward[i] -= self.ignored_warning_penalty

            # ============ 7. STILL IN CENTER (not trying): PENALTY ============
            still_in_center = in_center & ~current_in_room & ~excluded
            claiming_reward[still_in_center] -= self.center_penalty

            # ============ 8. PROGRESS SHAPING (moving toward rooms) ============
            if self.prev_potential is not None:
                dphi = potential - self.prev_potential  # positive = got closer
                # Only apply to agents not in a room
                not_in_room = ~current_in_room
                claiming_reward[not_in_room] += self.progress_scale * dphi[not_in_room]

            # ============ UPDATE STATE FOR NEXT STEP ============
            self.prev_in_room = current_in_room.clone()
            self.prev_room_idx = current_room_idx.clone()
            self.prev_room_was_full = current_room_full.clone()
            self.prev_target = feasible_target.clone()
            self.prev_potential = potential.clone()
            self.prev_winners = winners.clone()

            rewards += claiming_reward.unsqueeze(1)

            # Update target tracking in env
            if hasattr(env, 'prev_target_room'):
                env.prev_target_room = feasible_target.clone()

            # ============ GINI-BASED FAIRNESS PENALTY ============
            if hasattr(env, 'room_occupancy'):
                gini = self._compute_gini(env.room_occupancy.float())
            else:
                gini = self._compute_gini(occ.float())
            fairness_pen = -self.gini_penalty * gini
            rewards += fairness_pen

            # ============ SUCCESS BONUS ============
            if winners.all() and not excluded.any():
                rewards += self.success_bonus / n_agents
                if hasattr(env, 'success_achieved'):
                    env.success_achieved = True

        # ============ COLLISION AVOIDANCE ============
        agent_dists = torch.cdist(positions, positions)
        agent_dists.fill_diagonal_(float('inf'))
        min_agent_dist = agent_dists.min(dim=1).values

        min_safe_distance = 0.5
        too_close = min_agent_dist < min_safe_distance
        if too_close.any():
            closeness = (min_safe_distance - min_agent_dist[too_close]) / min_safe_distance
            collision_pen = torch.zeros(n_agents, device=device)
            collision_pen[too_close] = -self.collision_penalty * closeness
            rewards += collision_pen.unsqueeze(1)

        return rewards * self.reward_scale


class NormalizedRewardWrapper(RewardModule):
    """
    Wrapper that normalizes rewards from another module using running statistics.

    This helps stabilize training by keeping rewards in a consistent range
    regardless of the underlying reward scale.
    """
    def __init__(
        self,
        base_module: RewardModule,
        clip_range: float = 10.0,
        momentum: float = 0.99,
    ):
        super().__init__(phase_mode=base_module.phase_mode)
        self.base_module = base_module
        self.clip_range = clip_range
        self.momentum = momentum

        # Running statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def _activate(self):
        self.active = True
        self.base_module._activate()

    def _deactivate(self):
        self.active = False
        self.base_module._deactivate()

    def _reward(self, env: "MingleEnv") -> Tensor:
        raw_reward = self.base_module._reward(env)

        # Update running statistics
        batch_mean = raw_reward.mean().item()
        batch_var = raw_reward.var().item() if raw_reward.numel() > 1 else 1.0

        self.count += 1
        if self.count == 1:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        # Normalize
        std = max(self.running_var ** 0.5, 1e-8)
        normalized = (raw_reward - self.running_mean) / std

        # Clip to prevent extreme values
        return normalized.clamp(-self.clip_range, self.clip_range)
