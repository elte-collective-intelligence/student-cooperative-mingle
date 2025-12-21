# Communication Channel for Multi-Agent RL

This module extends the Mingle environment with **agent communication** capabilities, allowing agents to exchange information through:
- **Discrete symbolic messages** (fixed vocabulary)
- **Continuous learned embeddings** (rich representations)

## Features

- Drop-in replacement environments compatible with existing codebase
- Communication-aware policy networks
- Analysis tools for emergent communication patterns
- Standalone training scripts
- Easy integration with main system

## Quick Start

### 1. Train with Discrete Messages

```bash
# From project root
python3 -m communication_channel.train_comm
```

This trains agents that can send discrete messages from a vocabulary of 8 symbols.

### 2. Analyze Communication Patterns

```python
from communication_channel.envs import MingleEnvWithComm
from communication_channel.analysis import analyze_discrete_messages

# After training, analyze what the agents learned
results = analyze_discrete_messages(env, trained_policy, device, num_episodes=100)
```

## Project Structure

```
communication_channel/
├── envs/
│   ├── discrete_comm_env.py      # Discrete message environment
│   └── continuous_comm_env.py    # Continuous embedding environment
├── models/
│   ├── discrete_comm_policy.py   # Policy for discrete messages
│   └── continuous_comm_policy.py # Policy for continuous embeddings
├── analysis/
│   ├── message_analyzer.py       # Discrete message analysis
│   ├── embedding_analyzer.py     # Embedding analysis
│   └── comparison.py             # Compare w/ and w/o comm
├── configs/                      # Communication-specific configs
├── train_comm.py                 # Standalone training script
└── README.md                     # This file
```

## Implementation Details

### Discrete Communication (`MingleEnvWithComm`)

**How it works:**
- Agents output discrete message indices [0, vocab_size-1]
- Messages are one-hot encoded and added to observations
- Supports global or local (range-based) communication

**Observation space:**
- Base observations (14D) + received messages ((n_agents-1) × vocab_size)
- For 2 agents with vocab_size=8: 14 + 1×8 = 22 dimensions

**Action space:**
- Movement: (2D) continuous actions
- Message: (1D) discrete action ∈ {0, 1, ..., vocab_size-1}

**Example usage:**
```python
from communication_channel.envs import MingleEnvWithComm

env = MingleEnvWithComm(
    n_agents=4,
    vocab_size=8,        # 8 different messages
    comm_range=None,     # Global broadcast (or set float for local)
    # ... other MingleEnv parameters
)
```

### Continuous Communication (`MingleEnvWithEmbeddings`)

**How it works:**
- Agents output continuous embedding vectors
- Embeddings are aggregated (mean or attention)
- Richer information encoding than discrete messages

**Observation space:**
- Base observations (14D) + aggregated embeddings (embedding_dim)
- For embedding_dim=16: 14 + 16 = 30 dimensions

**Action space:**
- Movement: (2D) continuous actions
- Embedding: (embedding_dim) continuous values ∈ [-1, 1]

**Example usage:**
```python
from communication_channel.envs import MingleEnvWithEmbeddings

env = MingleEnvWithEmbeddings(
    n_agents=4,
    embedding_dim=16,
    aggregation="mean",  # or "attention"
    comm_range=5.0,      # Local communication radius
    # ... other MingleEnv parameters
)
```

## Analysis Tools

### Discrete Message Analysis

Analyzes:
- **Message frequency**: Which messages are used most?
- **Transition patterns**: How do messages change over time?
- **Reward correlation**: Which messages lead to higher rewards?
- **Message entropy**: Are agents using diverse communication?
- **Phase-specific usage**: Different messages for spinning vs claiming?

```python
from communication_channel.analysis import analyze_discrete_messages

results = analyze_discrete_messages(
    env=comm_env,
    policy=trained_policy,
    device=device,
    num_episodes=100,
    save_dir="communication_results/analysis"
)

# Access results
print(f"Message Entropy: {results['entropy']:.3f}")
print(f"Most used message: {results['counts'].argmax()}")
```

### Embedding Analysis

Analyzes:
- **PCA visualization**: Embedding space structure
- **Clustering**: Do embeddings form distinct groups?
- **Reward correlation**: Which embedding regions are beneficial?
- **State correlation**: What information is encoded?

```python
from communication_channel.analysis import analyze_embeddings

results = analyze_embeddings(
    env=embedding_env,
    policy=trained_policy,
    device=device,
    num_episodes=50
)
```

### Performance Comparison

Compare baseline (no communication) vs communication-enabled:

```python
from communication_channel.analysis import compare_performance

compare_performance({
    'baseline': baseline_logs,
    'discrete_comm': comm_logs,
    'continuous_comm': embedding_logs
})
```

## Integration with Main System

### Option 1: Use communication_channel standalone

The `communication_channel` folder is self-contained and can be run independently:

```bash
python3 -m communication_channel.train_comm
```

### Option 2: Integrate into main training pipeline

Modify `src/train/components.py`:

```python
# Add at top
from communication_channel.envs import MingleEnvWithComm
from communication_channel.models import build_discrete_comm_policy

# In build_train_components():
if config.get("communication", {}).get("enabled", False):
    env = MingleEnvWithComm(
        vocab_size=config["communication"]["vocab_size"],
        comm_range=config["communication"].get("comm_range", None),
        **env_params
    )
    policy = build_discrete_comm_policy(env, config["policy"], device)
else:
    # Existing code
    env = make_env(...)
    policy = build_policy(...)
```

### Option 3: Configuration-based switching

Add to `configs/env.yaml`:

```yaml
env:
  n_agents: 4
  # ... existing params

communication:
  enabled: true
  type: "discrete"  # or "continuous"
  vocab_size: 8
  embedding_dim: 16
  comm_range: null  # null = global, or float for local
  aggregation: "mean"  # for continuous only
```

## Configuration Parameters

### Discrete Communication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 8 | Number of discrete messages |
| `comm_range` | float\|None | None | Communication radius (None = global) |

### Continuous Communication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | 16 | Dimension of embedding vectors |
| `comm_range` | float\|None | None | Communication radius (None = global) |
| `aggregation` | str | "mean" | Aggregation method ("mean" or "attention") |

## Expected Results

### Training Performance

With communication, you should observe:
- **Faster convergence**: Agents coordinate more quickly
- **Higher final rewards**: Better teamwork and coordination
- **Emergent specialization**: Different messages for different contexts

### Message Usage Patterns

In the Mingle task:
- Messages during **spinning phase** may signal:
  - "I'm in the center" / "I'm outside"
  - "Move clockwise" / "Move counter-clockwise"
  - "Wait" / "Move now"

- Messages during **claiming phase** may signal:
  - "This room is full"
  - "Follow me to room X"
  - "I'm claiming this room"

### Typical Entropy

- **Low entropy (< 1.0)**: Agents converged to few messages (may indicate limited communication)
- **Medium entropy (1.0-2.0)**: Good balance of message diversity
- **High entropy (> 2.5)**: Agents use most messages (may not have converged to clear semantics)

## Troubleshooting

### Issue: Messages all the same

**Symptoms**: All agents send message 0 or one dominant message

**Solutions**:
- Increase training time - communication takes longer to learn
- Add entropy bonus to encourage exploration
- Try smaller vocabulary (4-6 messages instead of 8)

### Issue: No performance improvement

**Symptoms**: Communication model performs same as baseline

**Solutions**:
- Verify observation space includes messages (check dimensions)
- Ensure policy outputs message actions
- Try local communication instead of global (easier to learn)
- Increase number of agents (communication more valuable with more agents)

### Issue: Training unstable

**Symptoms**: NaN losses, diverging rewards

**Solutions**:
- Lower learning rate
- Check message one-hot encoding isn't causing dimension mismatches
- Verify action specs match environment expectations

## Examples

### Example 1: Quick test with 2 agents

```python
from communication_channel.envs import MingleEnvWithComm
from communication_channel.models import build_discrete_comm_policy

env = MingleEnvWithComm(n_agents=2, vocab_size=4)
# Train for a few batches to verify it works
```

### Example 2: Compare global vs local communication

```python
# Train two models
env_global = MingleEnvWithComm(n_agents=4, comm_range=None)
env_local = MingleEnvWithComm(n_agents=4, comm_range=3.0)

# Compare which learns better communication
```

### Example 3: Analyze emergent meaning

```python
# After training
results = analyze_discrete_messages(env, policy, device)

# Look at phase-specific usage
print("Spinning phase messages:", results['phase_usage']['spinning'])
print("Claiming phase messages:", results['phase_usage']['claiming'])

# Identify specialized messages
```

## Citation

If you use this communication module, please cite:

```bibtex
@misc{mingle_communication,
  title={Communication Channel for Multi-Agent Mingle Environment},
  author={Your Name},
  year={2025}
}
```

## Future Extensions

Potential improvements:
- **Addressed communication**: Send messages to specific agents
- **Multi-channel communication**: Different message types
- **Communication cost**: Penalty for sending messages
- **Theory of Mind**: Model what other agents know
- **Hierarchical messages**: Compositional semantics

## Support

For questions or issues:
1. Check this README
2. Review example scripts in `communication_channel/`
3. Open an issue in the repository

---

**Happy experimenting with emergent communication!** 🚀
