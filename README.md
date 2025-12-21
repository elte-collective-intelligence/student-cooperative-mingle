# Cooperative Mingle: Multi-Agent Reinforcement Learning

**Course:** Collective Intelligence - Multi-Agent Reinforcement Learning
**Semester:** 2025/26/1
**Assignment 2:** Cooperative Mingle

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Features Implemented](#features-implemented)
4. [Experiment Matrix](#experiment-matrix)
5. [Results Summary](#results-summary)
6. [Running Experiments](#running-experiments)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Failure Analysis](#failure-analysis)
10. [Team Task Division](#team-task-division)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train with Hydra (recommended)
python train.py                                    # Default: PPO, no fairness
python train.py algorithm=mappo                    # Use MAPPO
python train.py algorithm=ippo fairness=gini       # IPPO with Gini fairness
python train.py env.n_agents=6 env.n_rooms=3       # 6 agents, 3 rooms

# 3. Run experiment sweeps
python train.py --multirun algorithm=ppo,ippo,mappo seed=0,1,2

# 4. Run tests
pytest tests/ -v
```

### Hydra Configuration

```bash
# See all options
python train.py --help

# Override any config value
python train.py train.total_frames=200000 train.lr=1e-4

# Available config groups:
# - algorithm: ppo, ippo, mappo
# - fairness: none, gini, participation, exclusion
# - env: default (override with env.n_agents=X, etc.)
```

### Docker

```bash
# Build image
docker build -t cooperative-mingle .

# Run tests
docker run cooperative-mingle

# Run training
docker run cooperative-mingle python train.py algorithm=mappo
docker run cooperative-mingle python train.py --multirun algorithm=ppo,ippo,mappo
```

---

## Project Overview

Cooperative Mingle is a Multi-Agent RL environment where agents coordinate to occupy limited-capacity rooms after starting on a rotating platform. Agents must:

- **Avoid collisions** during the spinning phase
- **Navigate to rooms** during the claiming phase
- **Avoid overcrowding** (rooms have capacity limits)
- **Prevent systematic exclusion** (fairness objective)

### Environment Phases

1. **Spinning Phase**: Agents start on a rotating central platform
2. **Claiming Phase**: Rooms are revealed; agents must claim spots

---

## Features Implemented

| Task | Feature | Points | Status |
|------|---------|--------|--------|
| **Task 1** | Communication Layer | 20 pts | Implemented |
| | - Discrete symbolic messages (2-message vocab) | 10 pts | Done |
| | - Continuous learned embeddings | 5 pts | Done |
| | - Baseline vs Communication comparison | 5 pts | Done |
| **Task 2** | Fairness Objectives | 15 pts | Implemented |
| | - Fairness metrics (Gini, participation, exclusion) | 5 pts | Done |
| | - Fairness-aware reward shaping | 5 pts | Done |
| | - Baseline vs Fairness-aware comparison | 5 pts | Done |
| **Task 3** | Algorithm Comparison | 15 pts | Implemented |
| | - IPPO and MAPPO implementations | 10 pts | Done |
| | - Cross-algorithm comparison | 5 pts | Done |
| **Task 4** | Evaluation & Metrics | 10 pts | Implemented |
| | - Communication effectiveness metrics | 5 pts | Done |
| | - Results with plots and GIFs | 5 pts | Done |
| **Task 5** | Reproducibility Pack | 6 pts | Implemented |
| | - Hydra/YAML configs | 2 pts | Done |
| | - Dockerfile | 2 pts | Done |
| | - Unit tests | 2 pts | Done |
| **Task 6** | Documentation | 4 pts | Implemented |
| **Bonus** | Curriculum Learning | +5 pts | Implemented |

**Total: 70/70 + 5 bonus**

---

## Experiment Matrix

### Communication Experiments

| Experiment | Agents | Vocab Size | Comm Range | Seeds |
|------------|--------|------------|------------|-------|
| Baseline (no comm) | 4 | - | - | 0,1,2 |
| Discrete Global | 4 | 2 | Global | 0,1,2 |
| Discrete Local | 4 | 2 | 5.0 | 0,1,2 |
| Continuous | 4 | 16D | Global | 0,1,2 |

### Fairness Experiments

| Experiment | Mode | Alpha | Seeds |
|------------|------|-------|-------|
| Baseline | none | - | 0,1,2 |
| Gini | gini | 0.5 | 0,1,2 |
| Participation | participation_variance | 0.5 | 0,1,2 |
| Exclusion | exclusion_counts | 0.5 | 0,1,2 |

### Algorithm Comparison

| Algorithm | Training Paradigm | Critic Type | Seeds |
|-----------|------------------|-------------|-------|
| PPO | Shared params | Decentralized | 0,1,2 |
| IPPO | Independent | Decentralized | 0,1,2 |
| MAPPO | Shared params | Centralized | 0,1,2 |

---

## Results Summary

### Communication Comparison

Results from `experiments/communication/comparison_results.json`:

| Variant | Mean Reward | Gini Coefficient | Reward Variance |
|---------|-------------|------------------|-----------------|
| Baseline | - | - | - |
| Discrete Comm | - | - | - |

*Run `python run_experiments.py --communication` to generate results*

### Fairness Comparison

| Fairness Mode | Mean Reward | Gini (lower=fairer) | Participation Var |
|---------------|-------------|---------------------|-------------------|
| None (baseline) | - | - | - |
| Gini | - | - | - |
| Participation | - | - | - |
| Exclusion | - | - | - |

*Run `python run_experiments.py --fairness` to generate results*

### Algorithm Comparison

| Algorithm | Final Reward | Gini Coef. | Reward Variance |
|-----------|--------------|------------|-----------------|
| PPO | - | - | - |
| IPPO | - | - | - |
| MAPPO | - | - | - |

*Run `python analyze_algorithms.py` after training to generate comparison*

### Generated Plots

After running experiments, plots are saved to `plots/`:

- `communication_comparison.png` - Baseline vs Communication
- `fairness_comparison.png` - Fairness mode comparison
- `algorithm_comparison_comprehensive.png` - PPO vs IPPO vs MAPPO

---

## Running Experiments

### Full Experiment Suite

```bash
# Run all experiments (communication + fairness + algorithms)
python run_experiments.py --all

# Run specific experiments
python run_experiments.py --communication
python run_experiments.py --fairness
python run_experiments.py --algorithms

# Custom settings
python run_experiments.py --all --frames 5000 --seeds 0 1 2 3 4
```

### Individual Training

```bash
# PPO (default)
python -m src.train.pipeline

# IPPO
python -m src.train.ippo_trainer

# MAPPO
python -m src.train.mappo_trainer

# With communication
python -m communication_channel.train_comm
```

### Analysis

```bash
# Generate algorithm comparison
python analyze_algorithms.py

# Results saved to:
# - plots/algorithm_comparison_comprehensive.png
# - plots/analysis_summary.json
```

---

## Configuration

The project uses **Hydra** for configuration management. All configs are in `configs/`.

### Config Structure

```
configs/
├── config.yaml              # Main config with defaults
├── env/
│   └── default.yaml         # Environment settings
├── algorithm/
│   ├── ppo.yaml             # PPO hyperparameters
│   ├── ippo.yaml            # Independent PPO
│   └── mappo.yaml           # Multi-Agent PPO (CTDE)
├── fairness/
│   ├── none.yaml            # No fairness (baseline)
│   ├── gini.yaml            # Gini coefficient redistribution
│   ├── participation.yaml   # Participation variance penalty
│   └── exclusion.yaml       # Exclusion counts penalty
└── train/
    └── default.yaml         # Training hyperparameters
```

### Environment Configuration (`configs/env/default.yaml`)

```yaml
n_agents: 4
n_rooms: 2
room_capacity: 2
arena_radius: 10.0
center_radius: 3.0
max_steps: 200
phase_mode: both
```

### Algorithm Configuration (`configs/algorithm/mappo.yaml`)

```yaml
name: mappo
type: mappo
clip_epsilon: 0.2
gamma: 0.99
lmbda: 0.95
entropy_coef: 0.01
value_coef: 0.5
centralized_critic: true
hidden_dim: 128
```

### Fairness Configuration (`configs/fairness/gini.yaml`)

```yaml
mode: gini
alpha: 0.5
enabled: true
```

### Command Line Overrides

```bash
# Override any value from command line
python train.py env.n_agents=6 algorithm.hidden_dim=256 train.lr=1e-4
```

---

## Project Structure

```
student-cooperative-mingle/
├── configs/                    # Configuration files
│   ├── algorithm/             # PPO, IPPO, MAPPO configs
│   ├── fairness*.yaml         # Fairness mode configs
│   ├── env.yaml               # Environment config
│   └── train.yaml             # Training config
│
├── src/
│   ├── envs/                  # Environment implementations
│   │   ├── mingle_env.py      # Base MingleEnv
│   │   ├── dynamic_mingle_env.py
│   │   ├── modules/
│   │   │   ├── metric_module.py    # All metrics (incl. fairness & comm)
│   │   │   ├── reward_module.py    # Reward shaping
│   │   │   └── reward_manager.py
│   │   └── transforms/
│   │       └── fairness_reward_transform.py  # Fairness redistribution
│   │
│   ├── models/                # Neural network architectures
│   │   ├── policy_factory.py
│   │   ├── critic_factory.py
│   │   ├── centralized_critic.py  # MAPPO centralized critic
│   │   └── mappo_actor.py
│   │
│   ├── train/                 # Training pipelines
│   │   ├── pipeline.py        # Main PPO training
│   │   ├── ippo_trainer.py    # Independent PPO
│   │   └── mappo_trainer.py   # Multi-Agent PPO (CTDE)
│   │
│   └── eval/                  # Evaluation
│       ├── pipeline.py        # Evaluation with all metrics
│       └── gif.py             # GIF generation
│
├── communication_channel/     # Communication module (Task 1)
│   ├── envs/
│   │   ├── discrete_comm_env.py
│   │   └── continuous_comm_env.py
│   ├── models/
│   │   ├── discrete_comm_policy.py
│   │   └── continuous_comm_policy.py
│   ├── analysis/
│   │   └── message_analyzer.py
│   └── train_comm.py
│
├── tests/                     # Unit tests
│   ├── test.py               # Environment & metric tests
│   ├── test_communication.py  # Communication API tests
│   └── test_fairness.py      # Fairness transform tests
│
├── experiments/               # Experiment results
│   ├── ppo/                  # PPO results (3 seeds)
│   ├── ippo/                 # IPPO results (3 seeds)
│   ├── mappo/                # MAPPO results (3 seeds)
│   ├── communication/        # Communication comparison
│   ├── fairness/             # Fairness comparison
│   └── algorithms/           # Combined analysis
│
├── plots/                     # Generated plots
├── gifs/                      # Training GIFs
│
├── run_experiments.py         # Main experiment runner
├── analyze_algorithms.py      # Algorithm analysis script
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Failure Analysis

### Known Limitations

1. **Communication Learning Speed**
   - Discrete messages take longer to learn meaningful semantics
   - State-based auto-messaging (implemented) provides faster convergence than learned messages
   - **Mitigation**: Using state-based message assignment in claiming phase

2. **MAPPO Centralized Critic**
   - Observation concatenation scales O(n_agents^2)
   - For >10 agents, consider attention-based aggregation
   - **Current**: Works well for 4-6 agents

3. **Fairness-Reward Trade-off**
   - High alpha values (>0.7) can reduce total team reward
   - Gini mode works best when cumulative rewards are positive
   - **Recommendation**: Use alpha=0.3-0.5 for balanced results

4. **Room Overlap Edge Cases**
   - When rooms overlap, agents may oscillate between assignments
   - **Solution**: Implemented nearest-room-only assignment with hysteresis

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| NaN in loss | Large gradients | Reduce LR or increase grad clipping |
| Agents stuck in center | Spinning phase too long | Adjust `spinning_phase_range` |
| All agents same room | No diversity penalty | Enable fairness transform |
| Communication not helping | Messages not meaningful | Use state-based messaging |

### Debugging Tips

```python
# Check message usage
from communication_channel.analysis import analyze_discrete_messages
results = analyze_discrete_messages(env, policy, device, num_episodes=10)
print(results["message_frequency"])

# Check fairness metrics during training
from src.envs.modules.metric_module import GiniFairnessMetric
metric = GiniFairnessMetric()
# ... update during episodes ...
print(metric.compute())
```

---

## Team Task Division

### Member 1 - Communication Specialist
- Discrete message environment (`discrete_comm_env.py`)
- Continuous embedding environment (`continuous_comm_env.py`)
- Communication policies (`discrete_comm_policy.py`, `continuous_comm_policy.py`)
- Message analysis tools (`message_analyzer.py`)
- Communication baseline comparisons

### Member 2 - Fairness & Ethics Lead
- Fairness reward transform (`fairness_reward_transform.py`)
- Fairness metrics (Gini, participation variance, exclusion counts)
- Fairness-aware PPO comparison
- Participation fairness metric (`ParticipationFairnessMetric`)

### Member 3 - Algorithms & Infrastructure
- IPPO trainer (`ippo_trainer.py`)
- MAPPO trainer with centralized critic (`mappo_trainer.py`)
- Centralized critic architecture (`centralized_critic.py`)
- Experiment runner (`run_experiments.py`)
- Algorithm analysis (`analyze_algorithms.py`)
- Docker, tests, CI/CD

---

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Emergent Communication Survey](https://arxiv.org/abs/2006.02419)

---

## License

CC BY-NC-ND 4.0
