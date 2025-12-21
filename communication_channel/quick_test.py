"""
Quick test script to verify communication module works correctly.

This runs a minimal test to ensure all components are functioning.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication_channel.envs import MingleEnvWithComm
from communication_channel.models import build_discrete_comm_policy


def test_environment():
    """Test environment creation and basic functionality."""
    print("Testing environment...")

    env = MingleEnvWithComm(
        n_agents=2,
        vocab_size=4,
        comm_range=None
    )

    # Reset
    td = env.reset()
    print(f"  ✓ Environment reset successful")
    print(f"    - Observation shape: {td['observation'].shape}")
    print(f"    - Expected: (2, 18) [base 14 + 1*4 messages]")

    # Check action spec
    assert "action" in env.action_spec
    assert "message" in env.action_spec
    print(f"  ✓ Action spec correct: {list(env.action_spec.keys())}")

    # Create random actions
    actions = {
        "action": torch.randn(2, 2) * 0.1,
        "message": torch.randint(0, 4, (2,))
    }
    td.update(actions)

    # Step
    td = env.step(td)
    print(f"  ✓ Environment step successful")
    print(f"    - Reward shape: {td['next', 'reward'].shape}")

    return env


def test_policy(env):
    """Test policy creation and forward pass."""
    print("\nTesting policy...")

    # Minimal config
    policy_config = {
        "centralised": False,
        "share_params": True,
        "depth": 2,
        "num_cells": 64,
        "activation": "relu",
        "min_scale": 0.01
    }

    device = torch.device("cpu")
    policy = build_discrete_comm_policy(env, policy_config, device)

    print(f"  ✓ Policy created successfully")

    # Forward pass
    td = env.reset()
    with torch.no_grad():
        td_out = policy(td)

    print(f"  ✓ Policy forward pass successful")
    print(f"    - Output keys: {list(td_out.keys())}")
    assert "action" in td_out
    assert "message" in td_out
    print(f"  ✓ Policy outputs correct actions")

    return policy


def test_training_step(env, policy):
    """Test one training step."""
    print("\nTesting training step...")

    # Collect one rollout
    td = env.reset()
    trajectory = []

    for step in range(10):
        with torch.no_grad():
            td = policy(td)

        trajectory.append(td.clone())
        td = env.step(td)

        if td["terminated"].any():
            break

    print(f"  ✓ Collected {len(trajectory)} steps")
    print(f"  ✓ Training step simulation successful")


def main():
    print("\n" + "="*60)
    print("  COMMUNICATION MODULE QUICK TEST")
    print("="*60 + "\n")

    try:
        # Test environment
        env = test_environment()

        # Test policy
        policy = test_policy(env)

        # Test training step
        test_training_step(env, policy)

        print("\n" + "="*60)
        print("  ✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour communication module is ready to use!")
        print("\nNext steps:")
        print("  1. Run full training: python3 -m communication_channel.train_comm")
        print("  2. Check README: communication_channel/README.md")
        print("  3. Integration guide: communication_channel/INTEGRATION_GUIDE.md\n")

    except Exception as e:
        print("\n" + "="*60)
        print("  ❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nPlease check the error message above and verify your setup.\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
