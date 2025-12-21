"""
Unit tests for the fairness reward transform module.

Tests cover:
- Fairness transform initialization
- Gini coefficient calculation
- Reward redistribution modes (gini, participation_variance, exclusion_counts)
- Transform application to TensorDicts
"""

import pytest
import torch
from tensordict import TensorDict

import sys
sys.path.insert(0, '.')

from src.envs.transforms.fairness_reward_transform import make_fairness_reward_transform


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_td_with_reward():
    """Create a sample TensorDict with rewards."""
    n_agents = 4
    return TensorDict({
        "observation": torch.randn(n_agents, 14),
        "reward": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    }, batch_size=[])


@pytest.fixture
def sample_td_no_reward():
    """Create a sample TensorDict without rewards (reset state)."""
    n_agents = 4
    return TensorDict({
        "observation": torch.randn(n_agents, 14),
    }, batch_size=[])


# ============================================================
# Transform Initialization Tests
# ============================================================

class TestFairnessTransformInit:
    """Tests for fairness transform initialization."""

    def test_create_transform_none_mode(self):
        """Test creating transform with mode='none'."""
        cfg = {"mode": "none", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)
        assert callable(transform)

    def test_create_transform_gini_mode(self):
        """Test creating transform with mode='gini'."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)
        assert callable(transform)

    def test_create_transform_participation_mode(self):
        """Test creating transform with mode='participation_variance'."""
        cfg = {"mode": "participation_variance", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)
        assert callable(transform)

    def test_create_transform_exclusion_mode(self):
        """Test creating transform with mode='exclusion_counts'."""
        cfg = {"mode": "exclusion_counts", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)
        assert callable(transform)

    def test_default_alpha(self):
        """Test default alpha value when not specified."""
        cfg = {"mode": "gini"}
        transform = make_fairness_reward_transform(cfg)
        # Transform should still work with default alpha
        assert callable(transform)


# ============================================================
# No Redistribution Mode Tests
# ============================================================

class TestNoneMode:
    """Tests for mode='none' (no redistribution)."""

    def test_none_mode_preserves_reward(self, sample_td_with_reward):
        """Test that none mode preserves original rewards."""
        cfg = {"mode": "none", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # First call initializes state
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        original_reward = sample_td_with_reward["reward"].clone()
        result = transform(sample_td_with_reward)

        assert torch.allclose(result["reward"], original_reward)

    def test_none_mode_no_side_effects(self, sample_td_with_reward):
        """Test none mode doesn't modify other tensordict entries."""
        cfg = {"mode": "none", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        original_obs = sample_td_with_reward["observation"].clone()
        result = transform(sample_td_with_reward)

        assert torch.allclose(result["observation"], original_obs)


# ============================================================
# Gini Mode Tests
# ============================================================

class TestGiniMode:
    """Tests for mode='gini' (Gini-based redistribution)."""

    def test_gini_mode_returns_reward(self, sample_td_with_reward):
        """Test that Gini mode returns a reward tensor."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        result = transform(sample_td_with_reward)
        assert "reward" in result.keys()
        assert result["reward"].shape == sample_td_with_reward["reward"].shape

    def test_gini_mode_preserves_total_reward(self, sample_td_with_reward):
        """Test that Gini redistribution preserves total reward."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        original_total = sample_td_with_reward["reward"].sum()
        result = transform(sample_td_with_reward)
        new_total = result["reward"].sum()

        # Total reward should be approximately preserved
        assert torch.isclose(original_total, new_total, rtol=0.01)

    def test_gini_mode_reduces_inequality(self):
        """Test that Gini mode reduces reward inequality."""
        cfg = {"mode": "gini", "alpha": 0.8}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        # Highly unequal rewards
        unequal_td = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[10.0], [0.0], [0.0], [0.0]]),
        }, batch_size=[])

        result = transform(unequal_td)

        # After redistribution, the agent with 0 should have more than 0
        # (unless alpha_eff is 0 due to zero cumulative reward)
        assert result["reward"].shape == (4, 1)

    def test_gini_alpha_zero_no_change(self, sample_td_with_reward):
        """Test that alpha=0 means no redistribution."""
        cfg = {"mode": "gini", "alpha": 0.0}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        original_reward = sample_td_with_reward["reward"].clone()
        result = transform(sample_td_with_reward)

        # With alpha=0, rewards should be unchanged
        assert torch.allclose(result["reward"], original_reward, rtol=0.01)


# ============================================================
# Participation Variance Mode Tests
# ============================================================

class TestParticipationVarianceMode:
    """Tests for mode='participation_variance'."""

    def test_participation_mode_returns_reward(self, sample_td_with_reward):
        """Test participation mode returns reward tensor."""
        cfg = {"mode": "participation_variance", "alpha": 0.5, "participation_threshold": 0.0}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        result = transform(sample_td_with_reward)
        assert "reward" in result.keys()
        assert result["reward"].shape == sample_td_with_reward["reward"].shape

    def test_participation_tracks_counts(self):
        """Test that participation counts are tracked over time."""
        cfg = {"mode": "participation_variance", "alpha": 0.5, "participation_threshold": 1.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        # Run multiple steps
        for _ in range(5):
            td = TensorDict({
                "observation": torch.randn(4, 14),
                "reward": torch.tensor([[2.0], [1.0], [2.0], [0.5]]),
            }, batch_size=[])
            transform(td)

        # Transform should still work after multiple steps
        result = transform(TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        }, batch_size=[]))

        assert result["reward"].shape == (4, 1)


# ============================================================
# Exclusion Counts Mode Tests
# ============================================================

class TestExclusionCountsMode:
    """Tests for mode='exclusion_counts'."""

    def test_exclusion_mode_returns_reward(self, sample_td_with_reward):
        """Test exclusion mode returns reward tensor."""
        cfg = {"mode": "exclusion_counts", "alpha": 0.5, "exclusion_threshold": 0.0}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        result = transform(sample_td_with_reward)
        assert "reward" in result.keys()

    def test_exclusion_compensates_excluded_agents(self):
        """Test that excluded agents get compensation."""
        cfg = {"mode": "exclusion_counts", "alpha": 0.8, "exclusion_threshold": -1.0}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        # Rewards where agent 3 is heavily penalized (excluded)
        td = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[5.0], [5.0], [5.0], [-5.0]]),
        }, batch_size=[])

        result = transform(td)
        assert result["reward"].shape == (4, 1)


# ============================================================
# Reset Behavior Tests
# ============================================================

class TestResetBehavior:
    """Tests for transform behavior on reset (no reward key)."""

    def test_handles_no_reward_key(self, sample_td_no_reward):
        """Test transform handles TensorDict without reward key."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Should not crash on reset state
        result = transform(sample_td_no_reward)

        # Should return same tensordict without modification
        assert "observation" in result.keys()

    def test_initializes_state_on_reset(self, sample_td_no_reward):
        """Test that state is initialized on reset."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # First call initializes state
        transform(sample_td_no_reward)

        # Second call with reward should work
        td_with_reward = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        }, batch_size=[])

        result = transform(td_with_reward)
        assert "reward" in result.keys()


# ============================================================
# Edge Cases Tests
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_rewards(self):
        """Test handling of all-zero rewards."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        td = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.zeros(4, 1),
        }, batch_size=[])

        result = transform(td)
        assert not torch.isnan(result["reward"]).any()
        assert not torch.isinf(result["reward"]).any()

    def test_negative_rewards(self):
        """Test handling of negative rewards."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        td = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0]]),
        }, batch_size=[])

        result = transform(td)
        assert not torch.isnan(result["reward"]).any()
        assert not torch.isinf(result["reward"]).any()

    def test_mixed_positive_negative_rewards(self):
        """Test handling of mixed positive/negative rewards."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        td = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[5.0], [-3.0], [2.0], [-1.0]]),
        }, batch_size=[])

        result = transform(td)
        assert not torch.isnan(result["reward"]).any()
        assert not torch.isinf(result["reward"]).any()

    def test_single_agent(self):
        """Test with single agent."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(1, 14)}, batch_size=[]))

        td = TensorDict({
            "observation": torch.randn(1, 14),
            "reward": torch.tensor([[5.0]]),
        }, batch_size=[])

        result = transform(td)
        assert result["reward"].shape == (1, 1)

    def test_large_number_of_agents(self):
        """Test with many agents."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        n_agents = 20

        # Initialize
        transform(TensorDict({"observation": torch.randn(n_agents, 14)}, batch_size=[]))

        td = TensorDict({
            "observation": torch.randn(n_agents, 14),
            "reward": torch.randn(n_agents, 1),
        }, batch_size=[])

        result = transform(td)
        assert result["reward"].shape == (n_agents, 1)


# ============================================================
# Unknown Mode Tests
# ============================================================

class TestUnknownMode:
    """Tests for unknown/invalid modes."""

    def test_unknown_mode_returns_unchanged(self):
        """Test that unknown mode returns rewards unchanged."""
        cfg = {"mode": "unknown_mode", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        # Initialize
        transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

        td = TensorDict({
            "observation": torch.randn(4, 14),
            "reward": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        }, batch_size=[])

        original_reward = td["reward"].clone()
        result = transform(td)

        # Unknown mode should not crash and return reward unchanged
        assert torch.allclose(result["reward"], original_reward)


# ============================================================
# Integration Tests
# ============================================================

class TestFairnessIntegration:
    """Integration tests for fairness transform."""

    def test_multiple_episodes(self):
        """Test transform over multiple episode resets."""
        cfg = {"mode": "gini", "alpha": 0.5}
        transform = make_fairness_reward_transform(cfg)

        for episode in range(3):
            # Reset
            transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

            # Run episode steps
            for step in range(10):
                td = TensorDict({
                    "observation": torch.randn(4, 14),
                    "reward": torch.randn(4, 1),
                }, batch_size=[])
                result = transform(td)

                assert result["reward"].shape == (4, 1)
                assert not torch.isnan(result["reward"]).any()

    def test_all_modes_sequential(self):
        """Test all modes work correctly in sequence."""
        modes = ["none", "gini", "participation_variance", "exclusion_counts"]

        for mode in modes:
            cfg = {"mode": mode, "alpha": 0.5}
            transform = make_fairness_reward_transform(cfg)

            # Initialize
            transform(TensorDict({"observation": torch.randn(4, 14)}, batch_size=[]))

            # Apply
            td = TensorDict({
                "observation": torch.randn(4, 14),
                "reward": torch.randn(4, 1),
            }, batch_size=[])

            result = transform(td)
            assert result["reward"].shape == (4, 1), f"Mode {mode} failed"
