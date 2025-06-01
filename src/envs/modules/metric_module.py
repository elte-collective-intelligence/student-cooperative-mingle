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
    def __init__(self):
        super().__init__("room_switches")

    def reset(self):
        self.last_closest = None
        self.switch_count = 0

    def update(self, env: MingleEnv):
        if env.room_positions is None:
            return

        room_dists = torch.cdist(env.agent_positions, env.room_positions)
        closest = room_dists.argmin(dim=1)

        if self.last_closest is not None:
            switches = (closest != self.last_closest).sum().item()
            self.switch_count += switches

        self.last_closest = closest.clone()

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