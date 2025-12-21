# Integration Guide: Adding Communication to Your Main System

This guide shows you how to integrate the communication module with your existing training pipeline.

## Quick Integration (Recommended)

### Step 1: Run Communication Training Standalone

The easiest way is to use the standalone script:

```bash
# From project root
python3 -m communication_channel.train_comm
```

This runs completely independently from your main system and outputs results to `communication_results/`.

### Step 2: Compare Results

After training both baseline and communication models, compare:

```bash
# Baseline training (your existing system)
python3 -m src.train.pipeline

# Communication training
python3 -m communication_channel.train_comm

# Compare metrics in:
# - train_results/ (baseline)
# - communication_results/ (communication)
```

---

## Full Integration (Advanced)

If you want to integrate communication into your main training pipeline:

### Option A: Modify `src/train/components.py`

Add communication support to the existing components builder:

```python
# At the top of src/train/components.py
from typing import List, Optional
from communication_channel.envs import MingleEnvWithComm
from communication_channel.models import build_discrete_comm_policy

def build_train_components(
    config: dict,
    device: torch.device = torch.device("cpu"),
    reward_modules: Optional[List[RewardModule]] = None,
    reward_managers: Optional[dict] = None,
    use_communication: bool = False,  # NEW PARAMETER
    comm_config: Optional[dict] = None  # NEW PARAMETER
):
    # Environment selection
    if use_communication:
        print("🔊 Using communication-enabled environment")
        comm_config = comm_config or {}
        base_env = MingleEnvWithComm(
            n_agents=config["env"]["n_agents"],
            n_rooms=config["env"]["n_rooms"],
            arena_radius=config["env"]["arena_radius"],
            center_radius=config["env"]["center_radius"],
            max_steps=config["env"]["max_steps"],
            room_radius=config["env"]["room_radius"],
            room_capacity=config["env"]["room_capacity"],
            reward_modules=reward_modules,
            reward_managers=reward_managers,
            vocab_size=comm_config.get("vocab_size", 8),
            comm_range=comm_config.get("comm_range", None),
        )
    else:
        print("🔇 Using standard environment (no communication)")
        base_env = make_env(
            env_config=config["env"],
            device=device,
            reward_modules=reward_modules,
            reward_managers=reward_managers
        )

    env = TransformedEnv(
        base_env,
        Compose(ObservationNorm(in_keys=["observation"])),
    )
    env.reward_managers = reward_managers
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    env.to(device)

    # Policy selection
    if use_communication:
        policy = build_discrete_comm_policy(env, config["policy"], device)
    else:
        policy = build_policy(env, config["policy"], device)

    # Rest of the function remains the same...
    critic = build_critic(env, config["critic"], device)
    # ... etc
```

### Option B: Create Separate Config File

Create `configs/communication.yaml`:

```yaml
# Communication settings
communication:
  enabled: true
  type: "discrete"  # or "continuous"

  # Discrete message settings
  vocab_size: 8
  comm_range: null  # null for global, float for local

  # Continuous embedding settings (if type: continuous)
  embedding_dim: 16
  aggregation: "mean"  # or "attention"

# Use same env/policy/train configs as baseline
# (will be merged with other config files)
```

Then modify `src/train/pipeline.py`:

```python
if __name__ == "__main__":
    config = load_and_merge_configs("configs/")

    # Check if communication is enabled
    use_comm = config.get("communication", {}).get("enabled", False)

    if use_comm:
        from communication_channel.envs import MingleEnvWithComm
        from communication_channel.models import build_discrete_comm_policy
        # ... use communication components

    components = build_train_components(
        config,
        device,
        use_communication=use_comm,
        comm_config=config.get("communication", {})
    )
```

---

## Testing Your Integration

### 1. Verify Environment Creation

```python
from communication_channel.envs import MingleEnvWithComm

env = MingleEnvWithComm(n_agents=2, vocab_size=4)
td = env.reset()

print("Observation shape:", td["observation"].shape)
# Should be: (2, 14 + 4) = (2, 18)
# Base obs (14) + messages (4)

print("Action spec:", env.action_spec)
# Should have both "action" and "message" keys
```

### 2. Verify Policy Outputs

```python
from communication_channel.models import build_discrete_comm_policy

policy = build_discrete_comm_policy(env, config["policy"], device)
td_out = policy(td)

print("Output keys:", td_out.keys())
# Should include: "action", "message", "sample_log_prob"
```

### 3. Run a Quick Training Loop

```python
# Test 1 batch
for batch_data in collector:
    print("Collected batch with shape:", batch_data.batch_size)
    print("Keys:", batch_data.keys())
    # Should include message data
    break
```

---

## Comparison Workflow

### Suggested Experiment Setup

1. **Baseline (no communication)**
   ```bash
   python3 -m src.train.pipeline
   # Saves to: train_results/training_metrics_*/
   ```

2. **Discrete communication (global)**
   ```bash
   # Edit communication_channel/train_comm.py:
   # Set COMM_RANGE = None
   python3 -m communication_channel.train_comm
   # Saves to: communication_results/
   ```

3. **Discrete communication (local)**
   ```bash
   # Edit communication_channel/train_comm.py:
   # Set COMM_RANGE = 3.0
   python3 -m communication_channel.train_comm
   # Saves to: communication_results/
   ```

### Analysis Workflow

After training all variants:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open("train_results/.../metrics.json") as f:
    baseline = json.load(f)

with open("communication_results/metrics_comm.json") as f:
    comm = json.load(f)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(baseline['frames'], baseline['reward'], label='Baseline', alpha=0.7)
plt.plot(comm['frames'], comm['reward'], label='With Communication', alpha=0.7)
plt.xlabel('Frames')
plt.ylabel('Reward')
plt.title('Training Performance Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
baseline_final = np.mean(baseline['reward'][-10:])
comm_final = np.mean(comm['reward'][-10:])
plt.bar(['Baseline', 'Communication'], [baseline_final, comm_final])
plt.ylabel('Final Average Reward')
plt.title('Final Performance')
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('comparison.png')
```

---

## Troubleshooting Integration

### Issue: Import errors

**Error**: `ModuleNotFoundError: No module named 'communication_channel'`

**Fix**: Make sure you're running from the project root:
```bash
cd /path/to/student-cooperative-mingle
python3 -m communication_channel.train_comm
```

### Issue: Dimension mismatch

**Error**: `RuntimeError: shape mismatch` in observations

**Fix**: Check that observation spec updates correctly:
- Base env obs: 14D
- With messages (2 agents, vocab=8): 14 + 8 = 22D
- Formula: `14 + (n_agents - 1) * vocab_size`

### Issue: Action spec errors

**Error**: `KeyError: 'message'` when stepping environment

**Fix**: Ensure your policy outputs both actions:
```python
# Policy should return TensorDict with:
# - "action": movement actions
# - "message": discrete message indices
```

### Issue: Different performance but unclear why

**Troubleshooting steps**:
1. Verify same hyperparameters (lr, batch size, etc.)
2. Check observation dimensions are correct
3. Run message analysis to verify agents are communicating
4. Try increasing training time (communication takes longer to learn)

---

## Best Practices

### 1. Start Small
- Begin with 2 agents and vocab_size=4
- Verify everything works before scaling up
- Gradually increase complexity

### 2. Monitor Communication
- Use the analysis tools regularly
- Check message entropy during training
- Visualize message patterns

### 3. Hyperparameter Tuning
- Communication models may need:
  - Lower learning rates
  - More training frames
  - Higher entropy bonus initially

### 4. Baseline Comparison
- Always train baseline first
- Use same random seed for fair comparison
- Run multiple seeds to verify results

---

## Example Integration Script

Here's a complete example that runs both baseline and communication:

```python
#!/usr/bin/env python3
"""Compare baseline vs communication training."""

import torch
import sys
from src.utils.config import load_and_merge_configs
from src.train.pipeline import train
from src.train.components import build_train_components
from communication_channel.train_comm import build_comm_components

def main():
    config = load_and_merge_configs("configs/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline
    print("="*60)
    print("TRAINING BASELINE")
    print("="*60)
    baseline_components = build_train_components(config, device)
    baseline_logs = train(
        **baseline_components,
        device=device,
        total_frames=config["train"]["total_frames"],
        # ... other params
        metrics_save_path="results/baseline_metrics.json"
    )

    # Communication
    print("\n" + "="*60)
    print("TRAINING WITH COMMUNICATION")
    print("="*60)
    comm_components = build_comm_components(config, device)
    comm_logs = train(
        **comm_components,
        device=device,
        total_frames=config["train"]["total_frames"],
        # ... other params
        metrics_save_path="results/comm_metrics.json"
    )

    # Compare
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Baseline final reward: {baseline_logs['reward'][-1]:.3f}")
    print(f"Communication final reward: {comm_logs['reward'][-1]:.3f}")
    improvement = (comm_logs['reward'][-1] - baseline_logs['reward'][-1]) / abs(baseline_logs['reward'][-1]) * 100
    print(f"Improvement: {improvement:+.1f}%")

if __name__ == "__main__":
    main()
```

---

## Next Steps

After successful integration:

1. **Experiment with variants**:
   - Different vocabulary sizes
   - Local vs global communication
   - Continuous embeddings

2. **Analyze emergent semantics**:
   - What do different messages mean?
   - When are they used?
   - How do they correlate with performance?

3. **Extend the system**:
   - Add communication cost
   - Implement addressed communication
   - Try hierarchical messages

---

**Need help?** Check the main README or review the example scripts in `communication_channel/`.
