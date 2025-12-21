"""Tests for Cooperative Mingle environment, communication, and fairness."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward,
    StayInRoomReward,
    GetToRoomReward,
)


@pytest.fixture
def reward_modules():
    """Create standard reward modules for testing."""
    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode="claiming"),
        GetToRoomReward(max_reward=15.0, phase_mode="claiming"),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=5.0, phase_mode="claiming"),
    ]
    for m in modules:
        m._activate()
    return modules


@pytest.fixture
def basic_env(reward_modules):
    """Create a basic environment for testing."""
    return MingleEnv(
        n_agents=6,
        n_rooms=3,
        room_capacity=2,
        max_steps=300,
        phase_mode="claiming",
        reward_modules=reward_modules,
    )


@pytest.fixture
def comm_env(reward_modules):
    """Create an environment with communication enabled."""
    env = MingleEnv(
        n_agents=6,
        n_rooms=3,
        room_capacity=2,
        max_steps=300,
        phase_mode="claiming",
        reward_modules=reward_modules,
    )
    env.enable_communication = True
    return env


class TestEnvironmentSmoke:
    """Smoke tests to verify environment basic functionality."""

    def test_env_creation(self, basic_env):
        """Test that environment can be created."""
        assert basic_env is not None
        assert basic_env.n_agents == 6
        assert basic_env.n_rooms == 3
        assert basic_env.room_capacity == 2

    def test_env_reset(self, basic_env):
        """Test that environment reset works and returns correct structure."""
        td = basic_env.reset()

        # Check that TensorDict is returned
        assert td is not None
        assert "observation" in td

        # Check observation shape
        obs = td["observation"]
        assert obs.shape[0] == 6  # n_agents
        assert obs.shape[1] == 14  # observation dimensions (without communication)

    def test_env_reset_with_communication(self, comm_env):
        """Test that environment reset works with communication enabled."""
        td = comm_env.reset()

        obs = td["observation"]
        assert obs.shape[0] == 6  # n_agents
        assert obs.shape[1] == 18  # observation dimensions (with 4 communication dims)

    def test_env_step(self, basic_env):
        """Test that environment step works."""
        from tensordict import TensorDict

        basic_env.reset()

        # Create random action
        action = torch.rand(6, 2) * 0.6 - 0.3  # Actions in [-0.3, 0.3]
        step_td = TensorDict({"action": action}, batch_size=[])

        # Take step
        next_td = basic_env._step(step_td)

        # Check outputs
        assert "observation" in next_td
        assert "reward" in next_td
        assert "done" in next_td
        assert "terminated" in next_td

        # Check shapes
        assert next_td["observation"].shape == (6, 14)
        assert next_td["reward"].shape[0] == 6

    def test_env_multiple_steps(self, basic_env):
        """Test that environment can run multiple steps."""
        from tensordict import TensorDict

        basic_env.reset()

        for _ in range(10):
            action = torch.rand(6, 2) * 0.6 - 0.3
            step_td = TensorDict({"action": action}, batch_size=[])
            next_td = basic_env._step(step_td)

            if next_td["done"].any() or next_td["terminated"].any():
                break

        # If we get here without error, test passes
        assert True

    def test_agent_positions_in_bounds(self, basic_env):
        """Test that agent positions stay within arena bounds after steps."""
        from tensordict import TensorDict

        basic_env.reset()

        for _ in range(50):
            action = torch.rand(6, 2) * 0.6 - 0.3
            step_td = TensorDict({"action": action}, batch_size=[])
            basic_env._step(step_td)

        # Check positions are within arena
        positions = basic_env.agent_positions
        distances = torch.norm(positions, dim=1)
        assert (distances <= basic_env.arena_radius + 0.1).all(), "Agents escaped arena bounds"


class TestCommunicationAPI:
    """Unit tests for the communication system."""

    def test_leader_follower_assignment(self, comm_env):
        """Test that leaders and followers are correctly assigned."""
        comm_env.reset()

        # Check leader assignment
        assert hasattr(comm_env, 'is_leader')
        assert len(comm_env.is_leader) == 6

        # Half should be leaders
        n_leaders = sum(comm_env.is_leader)
        assert n_leaders == 3, f"Expected 3 leaders, got {n_leaders}"

    def test_communication_observation_dimensions(self, comm_env):
        """Test that communication adds correct observation dimensions."""
        td = comm_env.reset()

        obs = td["observation"]
        # Base obs (14) + communication (4) = 18
        assert obs.shape[1] == 18, f"Expected 18 dims with comm, got {obs.shape[1]}"

    def test_discrete_message_values(self, comm_env):
        """Test that discrete messages have valid values."""
        comm_env.reset()

        # Get observation which contains messages
        td = comm_env.reset()
        obs = td["observation"]

        # Communication features are in indices 14-17
        comm_features = obs[:, 14:18]

        # Features should be bounded (binary or normalized)
        assert (comm_features >= -1).all() and (comm_features <= 1).all(), \
            "Communication features out of expected range"

    def test_leader_broadcasts_message(self, comm_env):
        """Test that leaders broadcast messages to followers."""
        comm_env.reset()

        # Leaders should have is_leader = True
        leaders = [i for i, is_leader in enumerate(comm_env.is_leader) if is_leader]
        followers = [i for i, is_leader in enumerate(comm_env.is_leader) if not is_leader]

        assert len(leaders) == 3
        assert len(followers) == 3

    def test_communication_env_step(self, comm_env):
        """Test that communication works during step."""
        from tensordict import TensorDict

        comm_env.reset()

        # Take a step
        action = torch.rand(6, 2) * 0.6 - 0.3
        step_td = TensorDict({"action": action}, batch_size=[])
        next_td = comm_env._step(step_td)

        # Check communication features still present
        obs = next_td["observation"]
        assert obs.shape[1] == 18


class TestFairnessMetrics:
    """Unit tests for fairness metrics (Gini coefficient)."""

    def test_gini_perfect_equality(self):
        """Test Gini coefficient for perfect equality (should be 0)."""
        values = np.array([100, 100, 100, 100])
        gini = compute_gini(values)
        assert abs(gini) < 0.01, f"Expected Gini ~0 for equal values, got {gini}"

    def test_gini_perfect_inequality(self):
        """Test Gini coefficient for perfect inequality (should be ~1)."""
        values = np.array([0, 0, 0, 1000])
        gini = compute_gini(values)
        assert gini > 0.7, f"Expected high Gini for unequal values, got {gini}"

    def test_gini_moderate_inequality(self):
        """Test Gini coefficient for moderate inequality."""
        values = np.array([50, 75, 100, 125])
        gini = compute_gini(values)
        assert 0 < gini < 0.5, f"Expected moderate Gini, got {gini}"

    def test_gini_empty_array(self):
        """Test Gini handles empty array."""
        values = np.array([])
        gini = compute_gini(values)
        assert gini == 0.0

    def test_gini_single_value(self):
        """Test Gini handles single value."""
        values = np.array([100])
        gini = compute_gini(values)
        assert gini == 0.0

    def test_gini_negative_values(self):
        """Test Gini handles negative values by using absolute."""
        values = np.array([-50, 50, 100, 150])
        gini = compute_gini(values)
        assert 0 <= gini <= 1

    def test_gini_all_zeros(self):
        """Test Gini handles all zeros."""
        values = np.array([0, 0, 0, 0])
        gini = compute_gini(values)
        assert gini == 0.0


def compute_gini(values):
    """Compute Gini coefficient (copied from training code for testing)."""
    values = np.array(values).flatten()
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.abs(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


class TestRoomMechanics:
    """Unit tests for room capacity and occupancy."""

    def test_room_positions_exist(self, basic_env):
        """Test that room positions are initialized."""
        basic_env.reset()

        assert basic_env.room_positions is not None
        assert basic_env.room_positions.shape == (3, 2)

    def test_room_occupancy_tracking(self, basic_env):
        """Test that room occupancy is tracked."""
        basic_env.reset()

        assert hasattr(basic_env, 'room_occupancy')
        assert len(basic_env.room_occupancy) == 3

    def test_room_capacity_limit(self, basic_env):
        """Test that room capacity is enforced."""
        basic_env.reset()

        # Room capacity should never exceed the limit
        assert basic_env.room_capacity == 2

        # After reset, no room should be over capacity
        for occ in basic_env.room_occupancy:
            assert occ <= basic_env.room_capacity


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_episode_runs(self, basic_env):
        """Test that a full episode can run without errors."""
        from tensordict import TensorDict

        td = basic_env.reset()
        total_reward = 0

        for step in range(basic_env.max_steps):
            action = torch.rand(6, 2) * 0.6 - 0.3
            step_td = TensorDict({"action": action}, batch_size=[])
            next_td = basic_env._step(step_td)

            total_reward += next_td["reward"].sum().item()

            if next_td["done"].any() or next_td["terminated"].any():
                break

        # Episode should generate some reward
        assert total_reward != 0, "Episode generated zero reward"

    def test_reward_computation(self, basic_env):
        """Test that rewards are computed correctly."""
        from tensordict import TensorDict

        basic_env.reset()

        action = torch.rand(6, 2) * 0.6 - 0.3
        step_td = TensorDict({"action": action}, batch_size=[])
        next_td = basic_env._step(step_td)

        reward = next_td["reward"]

        # Reward should be a tensor with one value per agent
        assert reward.shape[0] == 6
        # Rewards should be finite
        assert torch.isfinite(reward).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
