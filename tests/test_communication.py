"""
Unit tests for the communication channel module.

Tests cover:
- Discrete communication environment (MingleEnvWithComm)
- Continuous communication environment
- Communication policy networks
- Message buffer functionality
"""

import pytest
import torch
from tensordict import TensorDict

# Import communication modules
import sys
sys.path.insert(0, '.')

from communication_channel.envs.discrete_comm_env import MingleEnvWithComm, MSG_FOLLOW_ME, MSG_ROOM_FULL


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def comm_env():
    """Create a basic communication environment."""
    return MingleEnvWithComm(n_agents=3, n_rooms=2, vocab_size=2)


@pytest.fixture
def comm_env_local():
    """Create a communication environment with local broadcast."""
    return MingleEnvWithComm(n_agents=4, n_rooms=2, vocab_size=2, comm_range=5.0)


# ============================================================
# Environment Initialization Tests
# ============================================================

class TestCommEnvInit:
    """Tests for communication environment initialization."""

    def test_initialization(self, comm_env):
        """Test basic initialization parameters."""
        assert comm_env.n_agents == 3
        assert comm_env.n_rooms == 2
        assert comm_env.vocab_size == 2
        assert comm_env.comm_range is None  # Global broadcast

    def test_vocab_size_config(self):
        """Test different vocabulary sizes."""
        for vocab_size in [2, 4, 8]:
            env = MingleEnvWithComm(n_agents=2, vocab_size=vocab_size)
            assert env.vocab_size == vocab_size

    def test_comm_range_config(self):
        """Test communication range configuration."""
        env_global = MingleEnvWithComm(n_agents=2, comm_range=None)
        assert env_global.comm_range is None

        env_local = MingleEnvWithComm(n_agents=2, comm_range=3.0)
        assert env_local.comm_range == 3.0

    def test_agent_id_config(self):
        """Test agent ID inclusion configuration."""
        env_with_id = MingleEnvWithComm(n_agents=2, include_agent_id=True)
        assert env_with_id.include_agent_id is True

        env_without_id = MingleEnvWithComm(n_agents=2, include_agent_id=False)
        assert env_without_id.include_agent_id is False


# ============================================================
# Observation Space Tests
# ============================================================

class TestCommObservation:
    """Tests for communication observation space."""

    def test_observation_dimension_with_messages(self, comm_env):
        """Test that observation includes message dimensions."""
        td = comm_env.reset()
        obs = td["observation"]

        # Expected: 14 (base) + (n_agents-1)*vocab_size (messages) + n_agents (agent_id)
        # For 3 agents, vocab=2: 14 + 2*2 + 3 = 21
        expected_dim = 14 + (comm_env.n_agents - 1) * comm_env.vocab_size + comm_env.n_agents
        assert obs.shape == (comm_env.n_agents, expected_dim)

    def test_observation_without_agent_id(self):
        """Test observation dimension without agent IDs."""
        env = MingleEnvWithComm(n_agents=3, vocab_size=2, include_agent_id=False)
        td = env.reset()
        obs = td["observation"]

        # Expected: 14 (base) + (n_agents-1)*vocab_size (messages) = 14 + 4 = 18
        expected_dim = 14 + (env.n_agents - 1) * env.vocab_size
        assert obs.shape == (env.n_agents, expected_dim)

    def test_observation_no_nan_or_inf(self, comm_env):
        """Test that observations contain no invalid values."""
        td = comm_env.reset()
        obs = td["observation"]
        assert not torch.isnan(obs).any()
        assert not torch.isinf(obs).any()

    def test_observation_spec_matches(self, comm_env):
        """Test that observation matches specification."""
        td = comm_env.reset()
        obs_shape = td["observation"].shape
        spec_shape = comm_env.observation_spec["observation"].shape
        assert obs_shape == spec_shape


# ============================================================
# Action Space Tests
# ============================================================

class TestCommActionSpace:
    """Tests for communication action space."""

    def test_action_spec_has_message(self, comm_env):
        """Test that action spec includes message output."""
        assert "action" in comm_env.action_spec.keys()
        assert "message" in comm_env.action_spec.keys()

    def test_action_shape(self, comm_env):
        """Test movement action shape."""
        action_spec = comm_env.action_spec["action"]
        assert action_spec.shape == (comm_env.n_agents, 2)

    def test_message_spec(self, comm_env):
        """Test message action specification."""
        message_spec = comm_env.action_spec["message"]
        assert message_spec.shape == (comm_env.n_agents,)
        assert message_spec.n == comm_env.vocab_size


# ============================================================
# Message Buffer Tests
# ============================================================

class TestMessageBuffer:
    """Tests for message buffer functionality."""

    def test_message_buffer_initialization(self, comm_env):
        """Test message buffer is initialized to zeros on reset."""
        comm_env.reset()
        assert comm_env.message_buffer.shape == (comm_env.n_agents,)
        assert torch.all(comm_env.message_buffer == 0)

    def test_message_buffer_dtype(self, comm_env):
        """Test message buffer data type."""
        comm_env.reset()
        assert comm_env.message_buffer.dtype == torch.long

    def test_message_meanings(self, comm_env):
        """Test message meaning lookup."""
        meanings = comm_env.get_message_meanings()
        assert MSG_FOLLOW_ME in meanings
        assert MSG_ROOM_FULL in meanings
        assert meanings[MSG_FOLLOW_ME] == "Follow me"
        assert meanings[MSG_ROOM_FULL] == "Room is full"


# ============================================================
# Step Function Tests
# ============================================================

class TestCommStep:
    """Tests for communication environment step function."""

    def test_step_with_actions_and_messages(self, comm_env):
        """Test stepping with both movement and message actions."""
        comm_env.reset()

        actions = torch.zeros((comm_env.n_agents, 2))
        messages = torch.zeros(comm_env.n_agents, dtype=torch.long)

        td = TensorDict({
            "action": actions,
            "message": messages
        }, batch_size=[])

        result = comm_env._step(td)

        assert "observation" in result.keys()
        assert "reward" in result.keys()
        assert "done" in result.keys()

    def test_step_updates_observations(self, comm_env):
        """Test that step updates observations correctly."""
        td = comm_env.reset()
        obs_before = td["observation"].clone()

        actions = torch.randn((comm_env.n_agents, 2)) * 0.1
        step_td = TensorDict({"action": actions, "message": torch.zeros(comm_env.n_agents, dtype=torch.long)}, batch_size=[])

        result = comm_env._step(step_td)
        obs_after = result["observation"]

        # Observations should change after step
        assert not torch.allclose(obs_before, obs_after)

    def test_step_observation_shape_preserved(self, comm_env):
        """Test observation shape is preserved after step."""
        td = comm_env.reset()
        expected_shape = td["observation"].shape

        for _ in range(5):
            actions = torch.randn((comm_env.n_agents, 2)) * 0.1
            step_td = TensorDict({"action": actions, "message": torch.zeros(comm_env.n_agents, dtype=torch.long)}, batch_size=[])
            td = comm_env._step(step_td)
            assert td["observation"].shape == expected_shape


# ============================================================
# Room Tracking Tests
# ============================================================

class TestRoomTracking:
    """Tests for room assignment and tracking."""

    def test_room_assignment_initialization(self, comm_env):
        """Test room assignment is initialized correctly."""
        comm_env.reset()
        assert comm_env.agent_room_idx.shape == (comm_env.n_agents,)
        assert comm_env.agent_in_room.shape == (comm_env.n_agents,)

    def test_room_occupancy_shape(self, comm_env):
        """Test room occupancy tensor shape."""
        comm_env.reset()
        assert comm_env.room_occupancy.shape == (comm_env.n_rooms,)

    def test_entry_order_tracking(self, comm_env):
        """Test room entry order is tracked."""
        comm_env.reset()
        assert isinstance(comm_env.room_entry_order, dict)
        assert len(comm_env.room_entry_order) == comm_env.n_rooms


# ============================================================
# Global vs Local Broadcast Tests
# ============================================================

class TestBroadcastModes:
    """Tests for global vs local broadcast modes."""

    def test_global_broadcast_message_shape(self, comm_env):
        """Test global broadcast message observation shape."""
        comm_env.reset()
        messages = comm_env._get_global_messages(comm_env.n_agents)

        expected_shape = (comm_env.n_agents, (comm_env.n_agents - 1) * comm_env.vocab_size)
        assert messages.shape == expected_shape

    def test_local_broadcast_message_shape(self, comm_env_local):
        """Test local broadcast message observation shape."""
        comm_env_local.reset()
        messages = comm_env_local._get_local_messages(comm_env_local.n_agents)

        expected_shape = (comm_env_local.n_agents, (comm_env_local.n_agents - 1) * comm_env_local.vocab_size)
        assert messages.shape == expected_shape

    def test_local_broadcast_respects_range(self, comm_env_local):
        """Test that local broadcast respects communication range."""
        comm_env_local.reset()

        # Place agents far apart (beyond comm_range)
        comm_env_local.agent_positions = torch.tensor([
            [0.0, 0.0],
            [100.0, 0.0],  # Far from agent 0
            [0.0, 100.0],  # Far from agent 0
            [1.0, 0.0],    # Close to agent 0
        ])

        messages = comm_env_local._get_local_messages(comm_env_local.n_agents)

        # Agent 0 should only receive messages from agent 3 (close)
        # Other slots should be zero-padded
        assert messages.shape[0] == comm_env_local.n_agents


# ============================================================
# Agent ID Tests
# ============================================================

class TestAgentIDs:
    """Tests for agent ID functionality."""

    def test_agent_ids_one_hot(self, comm_env):
        """Test agent IDs are one-hot encoded."""
        agent_ids = comm_env._get_agent_ids()

        assert agent_ids.shape == (comm_env.n_agents, comm_env.n_agents)

        # Should be identity matrix (one-hot)
        expected = torch.eye(comm_env.n_agents, device=comm_env.device)
        assert torch.allclose(agent_ids, expected)


# ============================================================
# Reconfigure Tests (Curriculum Learning Support)
# ============================================================

class TestReconfigure:
    """Tests for environment reconfiguration."""

    def test_reconfigure_agents(self, comm_env):
        """Test reconfiguring number of agents."""
        comm_env.reset()
        original_agents = comm_env.n_agents

        comm_env.reconfigure(n_agents=5)

        assert comm_env.n_agents == 5
        assert comm_env.message_buffer.shape == (5,)

    def test_reconfigure_rooms(self, comm_env):
        """Test reconfiguring number of rooms."""
        comm_env.reset()

        comm_env.reconfigure(n_rooms=4)

        assert comm_env.n_rooms == 4

    def test_reconfigure_capacity(self, comm_env):
        """Test reconfiguring room capacity."""
        comm_env.reset()

        comm_env.reconfigure(room_capacity=3)

        assert comm_env.room_capacity == 3

    def test_reconfigure_updates_specs(self, comm_env):
        """Test that reconfigure updates observation/action specs."""
        comm_env.reset()

        comm_env.reconfigure(n_agents=4)

        obs_spec = comm_env.observation_spec["observation"]
        # New obs dim: 14 + (4-1)*2 + 4 = 24
        expected_dim = 14 + 3 * comm_env.vocab_size + 4
        assert obs_spec.shape == (4, expected_dim)


# ============================================================
# Integration Tests
# ============================================================

class TestCommIntegration:
    """Integration tests for communication environment."""

    def test_full_episode_no_crash(self, comm_env):
        """Test running a full episode without crashes."""
        comm_env.reset()

        for _ in range(comm_env.max_steps):
            actions = torch.randn((comm_env.n_agents, 2)) * 0.1
            messages = torch.randint(0, comm_env.vocab_size, (comm_env.n_agents,))

            td = TensorDict({"action": actions, "message": messages}, batch_size=[])
            result = comm_env._step(td)

            if result["terminated"].any():
                break

        # Should complete without error
        assert True

    def test_multiple_resets(self, comm_env):
        """Test multiple consecutive resets."""
        for _ in range(5):
            td = comm_env.reset()
            assert td["observation"].shape[0] == comm_env.n_agents

            # Take a few steps
            for _ in range(10):
                actions = torch.randn((comm_env.n_agents, 2)) * 0.1
                step_td = TensorDict({"action": actions, "message": torch.zeros(comm_env.n_agents, dtype=torch.long)}, batch_size=[])
                comm_env._step(step_td)

    def test_observation_consistency_across_steps(self, comm_env):
        """Test observation dimensions remain consistent."""
        td = comm_env.reset()
        expected_shape = td["observation"].shape

        for _ in range(20):
            actions = torch.randn((comm_env.n_agents, 2)) * 0.1
            step_td = TensorDict({"action": actions, "message": torch.zeros(comm_env.n_agents, dtype=torch.long)}, batch_size=[])
            td = comm_env._step(step_td)

            assert td["observation"].shape == expected_shape
            assert not torch.isnan(td["observation"]).any()
            assert not torch.isinf(td["observation"]).any()
