import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.envs.mingle_env import MingleEnv


class MetricModule(ABC):
    """Abstract base class for custom environment metrics."""
    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def update(self, env: MingleEnv):
        """Update internal metric state given the current environment state."""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Compute final metric value(s)."""
        pass

    def reset(self):
        """Reset internal tracking variables."""
        pass

class CollisionRateMetric(MetricModule):
    def __init__(self, min_distance: float = 0.5):
        super().__init__("collision_rate")
        self.min_distance = min_distance

    def reset(self):
        self.total_collisions = 0
        self.total_checks = 0

    def update(self, env: MingleEnv):
        dists = torch.cdist(env.agent_positions, env.agent_positions)
        mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()  # ignore self and duplicates
        close_pairs = (dists < self.min_distance) & mask

        self.total_collisions += close_pairs.sum().item()
        self.total_checks += mask.sum().item()

    def compute(self) -> Dict[str, float]:
        rate = self.total_collisions / max(self.total_checks, 1)
        return {self.name: rate}

class RoomOccupancyRateMetric(MetricModule):
    def __init__(self):
        super().__init__("room_occupancy_rate")

    def reset(self):
        self.agents_in_rooms = 0
        self.total_agents = 0

    def update(self, env: MingleEnv):
        if env.room_positions is None:
            return

        dists = torch.cdist(env.agent_positions, env.room_positions)

        in_any_room = (dists < env.room_radius).any(dim=1)

        self.agents_in_rooms += in_any_room.sum().item()
        self.total_agents += env.n_agents

    def compute(self) -> Dict[str, float]:
        rate = self.agents_in_rooms / max(self.total_agents, 1)
        return {self.name: rate}
    
class CenterPresenceMetric(MetricModule):
    def __init__(self):
        super().__init__("center_presence_rate")

    def reset(self):
        self.center_steps = 0
        self.total_steps = 0

    def update(self, env: MingleEnv):
        in_center = env.agent_positions.norm(dim=1) <= env.center_radius
        self.center_steps += in_center.sum().item()
        self.total_steps += env.n_agents

    def compute(self) -> Dict[str, float]:
        rate = self.center_steps / max(self.total_steps, 1)
        return {self.name: rate}

class AverageStepDistanceMetric(MetricModule):
    def __init__(self):
        super().__init__("average_step_distance")

    def reset(self):
        self.total_distance = 0.0
        self.step_count = 0
        self.last_positions = None

    def update(self, env: MingleEnv):
        if self.last_positions is not None:
            distances = (env.agent_positions - self.last_positions).norm(dim=1)
            self.total_distance += distances.sum().item()
            self.step_count += env.n_agents
        self.last_positions = env.agent_positions.clone()

    def compute(self) -> Dict[str, float]:
        return {
            self.name: self.total_distance / max(self.step_count, 1)
        }

class IdleAgentRateMetric(MetricModule):
    def __init__(self, threshold: float = 0.01):
        super().__init__("idle_agent_rate")
        self.threshold = threshold

    def reset(self):
        self.idle_steps = 0
        self.total_steps = 0
        self.last_positions = None

    def update(self, env: MingleEnv):
        if self.last_positions is not None:
            movement = (env.agent_positions - self.last_positions).norm(dim=1)
            idle = (movement < self.threshold).sum().item()
            self.idle_steps += idle
            self.total_steps += env.n_agents
        self.last_positions = env.agent_positions.clone()

    def compute(self) -> Dict[str, float]:
        return {self.name: self.idle_steps / max(self.total_steps, 1)}

class RoomSwitchesMetric(MetricModule):
    """
    Counts target room switches during CLAIMING phase only.
    Uses feasibility-aware targeting (excludes full rooms) with hysteresis.
    """
    def __init__(self, hysteresis_margin: float = 0.3):
        super().__init__("room_switches")
        self.hysteresis_margin = hysteresis_margin

    def reset(self):
        self.last_target = None
        self.switch_count = 0

    def _get_feasible_target(self, env: MingleEnv) -> torch.Tensor:
        """Get nearest feasible room for each agent (excludes full rooms)."""
        room_dists = torch.cdist(env.agent_positions, env.room_positions)
        min_room_dist = room_dists.min(dim=1).values
        closest_room = room_dists.argmin(dim=1)

        # Compute room occupancy
        in_room = min_room_dist < env.room_radius
        room_occupancy = torch.zeros(env.n_rooms, device=env.device)
        for i in range(env.n_agents):
            if in_room[i]:
                room_occupancy[closest_room[i]] += 1

        # Feasibility mask: room not full, OR agent already in it
        n_agents = env.n_agents
        n_rooms = env.n_rooms
        feasible_mask = torch.zeros(n_agents, n_rooms, dtype=torch.bool, device=env.device)
        for i in range(n_agents):
            for r in range(n_rooms):
                is_full = room_occupancy[r] >= env.room_capacity
                agent_in_this_room = in_room[i] and closest_room[i] == r
                feasible_mask[i, r] = (not is_full) or agent_in_this_room

        # Fallback if no feasible
        no_feasible = ~feasible_mask.any(dim=1)
        feasible_mask[no_feasible] = True

        # Masked distances
        masked_dists = room_dists.clone()
        masked_dists[~feasible_mask] = float('inf')

        _, feasible_target = masked_dists.min(dim=1)
        return feasible_target, masked_dists

    def update(self, env: MingleEnv):
        # ONLY count during claiming phase
        if env.phase != "claiming":
            self.last_target = None
            return

        if env.room_positions is None:
            return

        feasible_target, masked_dists = self._get_feasible_target(env)

        if self.last_target is not None:
            # Hysteresis: only count switch if new target is closer by margin
            for i in range(env.n_agents):
                if feasible_target[i] != self.last_target[i]:
                    old_dist = masked_dists[i, self.last_target[i]]
                    new_dist = masked_dists[i, feasible_target[i]]
                    # Only count if genuinely switching (new is significantly closer)
                    if new_dist < old_dist - self.hysteresis_margin:
                        self.switch_count += 1

        self.last_target = feasible_target.clone()

    def compute(self) -> Dict[str, float]:
        return {self.name: self.switch_count}

class PhaseTimeMetric(MetricModule):
    def __init__(self):
        super().__init__("phase_time_distribution")

    def reset(self):
        self.phase_counts = {"spinning": 0, "claiming": 0}
        self.total = 0

    def update(self, env: MingleEnv):
        if env.phase in self.phase_counts:
            self.phase_counts[env.phase] += 1
        self.total += 1

    def compute(self) -> Dict[str, float]:
        return {
            f"phase_time_{phase}": count / max(self.total, 1)
            for phase, count in self.phase_counts.items()
        }

class AgentDensityMetric(MetricModule):
    def __init__(self, radius: float = 1.0):
        super().__init__("agent_density")
        self.radius = radius
        self.total_density = 0.0
        self.total_steps = 0

    def reset(self):
        self.total_density = 0.0
        self.total_steps = 0

    def update(self, env: MingleEnv):
        dists = torch.cdist(env.agent_positions, env.agent_positions)
        mask = (dists < self.radius) & ~torch.eye(env.n_agents, dtype=torch.bool, device=env.device)
        close_counts = mask.sum(dim=1).float().mean().item()
        self.total_density += close_counts
        self.total_steps += 1

    def compute(self) -> Dict[str, float]:
        return {self.name: self.total_density / max(self.total_steps, 1)}

class MaxDistanceFromCenterMetric(MetricModule):
    def __init__(self):
        super().__init__("max_distance_from_center")
        self.max_distance = 0.0

    def reset(self):
        self.max_distance = 0.0

    def update(self, env: MingleEnv):
        distances = env.agent_positions.norm(dim=1)
        self.max_distance = max(self.max_distance, distances.max().item())

    def compute(self):
        return {self.name: self.max_distance}

class MinAgentDistanceMetric(MetricModule):
    def __init__(self):
        super().__init__("min_agent_distance")
        self.min_distance = float('inf')

    def reset(self):
        self.min_distance = float('inf')

    def update(self, env: MingleEnv):
        dists = torch.cdist(env.agent_positions, env.agent_positions)
        mask = ~torch.eye(env.n_agents, dtype=torch.bool)
        min_dist = dists[mask].min().item()
        self.min_distance = min(self.min_distance, min_dist)

    def compute(self):
        return {self.name: self.min_distance if self.min_distance != float('inf') else 0.0}

class AverageRoomDistanceMetric(MetricModule):
    def __init__(self):
        super().__init__("average_room_distance")

    def reset(self):
        self.total_distance = 0.0
        self.count = 0

    def update(self, env: MingleEnv):
        if env.room_positions is None:
            return
        dists = torch.cdist(env.agent_positions, env.room_positions)
        closest = dists.min(dim=1)[0]
        self.total_distance += closest.sum().item()
        self.count += env.n_agents

    def compute(self):
        return {self.name: self.total_distance / max(self.count, 1)}

class AgentMovementVarianceMetric(MetricModule):
    def __init__(self):
        super().__init__("agent_movement_variance")
        self.positions_history = []

    def reset(self):
        self.positions_history = []

    def update(self, env: MingleEnv):
        self.positions_history.append(env.agent_positions.clone())

    def compute(self):
        if len(self.positions_history) < 2:
            return {self.name: 0.0}
        diffs = [
            (self.positions_history[i+1] - self.positions_history[i]).norm(dim=1)
            for i in range(len(self.positions_history)-1)
        ]
        all_moves = torch.cat(diffs)
        variance = all_moves.var().item()
        return {self.name: variance}


class EpisodeSuccessMetric(MetricModule):
    """
    Tracks whether all agents are successfully placed in rooms within capacity.

    Success = all agents in rooms AND no room overcrowded.
    """
    def __init__(self):
        super().__init__("episode_success")

    def reset(self):
        self.success_achieved = False
        self.success_step = -1
        self.final_agents_in_valid_rooms = 0
        self.total_agents = 0

    def update(self, env: MingleEnv):
        if env.room_positions is None or env.phase != "claiming":
            return

        self.total_agents = env.n_agents

        # Check room occupancy
        room_dists = torch.cdist(env.agent_positions, env.room_positions)

        # Count agents in each room
        room_occupancy = torch.zeros(env.n_rooms, device=env.device)
        agent_room_assignment = torch.full((env.n_agents,), -1, dtype=torch.long, device=env.device)

        for i in range(env.n_agents):
            in_room_mask = room_dists[i] < env.room_radius
            if in_room_mask.any():
                room_idx = in_room_mask.nonzero(as_tuple=True)[0][0].item()
                agent_room_assignment[i] = room_idx
                room_occupancy[room_idx] += 1

        # Count agents in valid (non-overcrowded) rooms
        agents_in_valid_rooms = 0
        for i in range(env.n_agents):
            room_idx = agent_room_assignment[i].item()
            if room_idx >= 0 and room_occupancy[room_idx] <= env.room_capacity:
                agents_in_valid_rooms += 1

        self.final_agents_in_valid_rooms = agents_in_valid_rooms

        # Success: all agents in valid rooms
        if agents_in_valid_rooms == env.n_agents and not self.success_achieved:
            self.success_achieved = True
            self.success_step = env.current_step

    def compute(self):
        success_rate = self.final_agents_in_valid_rooms / max(self.total_agents, 1)
        return {
            "episode_success": 1.0 if self.success_achieved else 0.0,
            "agents_in_valid_rooms_rate": success_rate,
            "success_step": self.success_step if self.success_achieved else -1,
        }


class GiniFairnessMetric(MetricModule):
    """
    Tracks Gini coefficient for reward fairness.

    Lower Gini = more equal distribution of rewards.
    Gini of 0 = perfect equality, Gini of 1 = maximum inequality.
    """
    def __init__(self):
        super().__init__("gini_fairness")

    def reset(self):
        self.cumulative_rewards = None

    def update(self, env: MingleEnv):
        if hasattr(env, 'last_rewards'):
            rewards = env.last_rewards.squeeze()
            if self.cumulative_rewards is None:
                self.cumulative_rewards = rewards.clone()
            else:
                self.cumulative_rewards += rewards

    def compute(self):
        if self.cumulative_rewards is None:
            return {"gini_coefficient": 0.0, "reward_variance": 0.0}

        rewards = self.cumulative_rewards
        n = len(rewards)

        # Gini coefficient calculation
        sorted_rewards, _ = torch.sort(rewards)
        cumsum = torch.cumsum(sorted_rewards, dim=0)
        sum_rewards = rewards.sum()

        if sum_rewards == 0:
            gini = 0.0
        else:
            # Gini = 1 - 2 * (sum of cumulative / (n * total))
            gini = 1 - 2 * cumsum.sum() / (n * sum_rewards) + 1 / n
            gini = max(0.0, min(1.0, gini.item()))

        variance = rewards.var().item()

        return {
            "gini_coefficient": gini,
            "reward_variance": variance,
            "reward_range": (rewards.max() - rewards.min()).item(),
        }


class ClaimingPhaseEfficiencyMetric(MetricModule):
    """
    Tracks how efficiently agents leave center and enter rooms during claiming.
    """
    def __init__(self):
        super().__init__("claiming_efficiency")

    def reset(self):
        self.claiming_steps = 0
        self.center_steps_during_claiming = 0
        self.room_steps_during_claiming = 0

    def update(self, env: MingleEnv):
        if env.phase != "claiming":
            return

        self.claiming_steps += 1

        # Count agents in center vs rooms
        center_dists = env.agent_positions.norm(dim=1)
        in_center = center_dists < env.center_radius
        self.center_steps_during_claiming += in_center.sum().item()

        if env.room_positions is not None:
            room_dists = torch.cdist(env.agent_positions, env.room_positions)
            in_room = (room_dists < env.room_radius).any(dim=1)
            self.room_steps_during_claiming += in_room.sum().item()

    def compute(self):
        total_agent_steps = self.claiming_steps * 4  # Assuming 4 agents, adjust if needed
        if total_agent_steps == 0:
            return {
                "center_presence_in_claiming": 0.0,
                "room_presence_in_claiming": 0.0,
            }

        return {
            "center_presence_in_claiming": self.center_steps_during_claiming / total_agent_steps,
            "room_presence_in_claiming": self.room_steps_during_claiming / total_agent_steps,
        }


# ============================================================
# Communication Effectiveness Metrics
# ============================================================

class MessageUsageMetric(MetricModule):
    """
    Tracks message usage patterns in communication-enabled environments.

    Measures:
    - Message frequency distribution
    - Message entropy (diversity)
    - Message-action correlation
    """
    def __init__(self):
        super().__init__("message_usage")

    def reset(self):
        self.message_counts = {}
        self.total_messages = 0
        self.message_room_correlation = []

    def update(self, env):
        # Check if environment has communication
        if not hasattr(env, 'message_buffer'):
            return

        messages = env.message_buffer
        for msg in messages.tolist():
            self.message_counts[msg] = self.message_counts.get(msg, 0) + 1
            self.total_messages += 1

        # Track correlation between messages and room status
        if hasattr(env, 'agent_in_room'):
            for i, msg in enumerate(messages.tolist()):
                in_room = env.agent_in_room[i].item() if hasattr(env.agent_in_room[i], 'item') else env.agent_in_room[i]
                self.message_room_correlation.append((msg, in_room))

    def compute(self):
        if self.total_messages == 0:
            return {
                "message_entropy": 0.0,
                "message_diversity": 0.0,
                "dominant_message_ratio": 0.0,
            }

        import math

        # Calculate entropy
        probs = [count / self.total_messages for count in self.message_counts.values()]
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)

        # Max entropy for uniform distribution
        n_messages = len(self.message_counts)
        max_entropy = math.log2(n_messages) if n_messages > 1 else 1.0

        # Normalized entropy (diversity)
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        # Dominant message ratio
        max_count = max(self.message_counts.values()) if self.message_counts else 0
        dominant_ratio = max_count / self.total_messages

        # Message-room correlation (how often "room full" is sent when in room)
        correlation = 0.0
        if self.message_room_correlation:
            # Message 1 typically means "room full"
            room_full_in_room = sum(1 for msg, in_room in self.message_room_correlation if msg == 1 and in_room)
            room_full_total = sum(1 for msg, _ in self.message_room_correlation if msg == 1)
            if room_full_total > 0:
                correlation = room_full_in_room / room_full_total

        return {
            "message_entropy": entropy,
            "message_diversity": diversity,
            "dominant_message_ratio": dominant_ratio,
            "message_room_correlation": correlation,
        }


class CommunicationEffectivenessMetric(MetricModule):
    """
    Measures whether communication leads to better coordination.

    Tracks:
    - Response to "room full" warnings
    - Coordination success rate
    - Message timing relative to room entry
    """
    def __init__(self):
        super().__init__("communication_effectiveness")

    def reset(self):
        self.warnings_heeded = 0
        self.warnings_ignored = 0
        self.successful_coordinations = 0
        self.total_room_attempts = 0

    def update(self, env):
        # Check if environment has communication
        if not hasattr(env, 'message_buffer') or not hasattr(env, 'agent_in_room'):
            return

        # Track room attempts and warnings
        if hasattr(env, 'agent_excluded'):
            excluded = env.agent_excluded
            for i in range(env.n_agents):
                if excluded[i]:
                    self.total_room_attempts += 1
                    # Check if agent was warned
                    if hasattr(env, 'received_room_full_warning'):
                        warnings = env.received_room_full_warning.get(i, set())
                        if hasattr(env, 'excluded_room_idx'):
                            excluded_room = env.excluded_room_idx[i].item()
                            if excluded_room in warnings:
                                self.warnings_ignored += 1

        # Track successful room entries
        if hasattr(env, 'agent_in_room'):
            in_room = env.agent_in_room
            for i in range(env.n_agents):
                if in_room[i]:
                    room_idx = env.agent_room_idx[i].item() if hasattr(env, 'agent_room_idx') else -1
                    if room_idx >= 0 and env.room_occupancy[room_idx] <= env.room_capacity:
                        self.successful_coordinations += 1

    def compute(self):
        warning_compliance = 1.0 - (self.warnings_ignored / max(self.total_room_attempts, 1))

        return {
            "warning_compliance_rate": warning_compliance,
            "warnings_ignored": self.warnings_ignored,
            "total_room_attempts": self.total_room_attempts,
            "successful_coordinations": self.successful_coordinations,
        }


class ParticipationFairnessMetric(MetricModule):
    """
    Tracks participation fairness - how evenly agents participate in room claiming.

    Measures:
    - Per-agent room time
    - Participation variance
    - Exclusion counts per agent
    """
    def __init__(self):
        super().__init__("participation_fairness")

    def reset(self):
        self.agent_room_time = None
        self.agent_exclusion_counts = None
        self.total_steps = 0

    def update(self, env):
        if self.agent_room_time is None:
            self.agent_room_time = torch.zeros(env.n_agents)
            self.agent_exclusion_counts = torch.zeros(env.n_agents)

        # Track room time
        if hasattr(env, 'agent_in_room'):
            self.agent_room_time += env.agent_in_room.float().cpu()

        # Track exclusions
        if hasattr(env, 'agent_excluded'):
            self.agent_exclusion_counts += env.agent_excluded.float().cpu()

        self.total_steps += 1

    def compute(self):
        if self.agent_room_time is None or self.total_steps == 0:
            return {
                "participation_variance": 0.0,
                "exclusion_variance": 0.0,
                "participation_gini": 0.0,
            }

        # Participation rate per agent
        participation_rates = self.agent_room_time / self.total_steps

        # Variance in participation
        participation_variance = participation_rates.var().item()

        # Exclusion variance
        exclusion_variance = self.agent_exclusion_counts.var().item()

        # Gini coefficient for participation
        sorted_rates, _ = torch.sort(participation_rates)
        n = len(sorted_rates)
        cumsum = torch.cumsum(sorted_rates, dim=0)
        total = sorted_rates.sum()

        if total > 0:
            gini = 1 - 2 * cumsum.sum() / (n * total) + 1 / n
            gini = max(0.0, min(1.0, gini.item()))
        else:
            gini = 0.0

        return {
            "participation_variance": participation_variance,
            "exclusion_variance": exclusion_variance,
            "participation_gini": gini,
            "mean_participation_rate": participation_rates.mean().item(),
            "max_exclusions": self.agent_exclusion_counts.max().item(),
        }