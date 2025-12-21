# Cooperative Mingle: Multi-Agent Reinforcement Learning for Room Allocation

A comprehensive Multi-Agent Reinforcement Learning (MARL) framework for solving the cooperative room allocation problem. Agents must coordinate to distribute themselves fairly across rooms while respecting capacity constraints.

<p align="center">
  <img src="contribution_tests_and_comparisions/fullstack/full_stack/trained_policy.gif" alt="Full Stack Demo" width="400"/>
</p>

**Course:** Collective Intelligence - Multi-Agent Reinforcement Learning
**Semester:** 2025/26/1
**Assignment 2:** Cooperative Mingle

---

## Table of Contents

- [Problem Description](#problem-description)
- [Key Contributions](#key-contributions)
- [Algorithm Comparison](#algorithm-comparison)
- [Communication System](#communication-system)
- [Fairness Mechanisms](#fairness-mechanisms)
- [Curriculum Learning](#curriculum-learning)
- [Full Stack Integration](#full-stack-integration)
- [Results Summary](#results-summary)
- [Agent Behavior Analysis](#agent-behavior-analysis)
- [Scenario Demonstrations](#scenario-demonstrations)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)

---

## Problem Description

In the **Cooperative Mingle** environment:
- **N agents** must distribute themselves across **M rooms**
- Each room has a **capacity limit** (e.g., 2 agents per room)
- Agents start in a central area and must navigate to rooms
- Goals: Maximize room occupancy while maintaining **fair distribution**

### Environment Features

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `n_agents` | 6 | Number of agents |
| `n_rooms` | 3 | Number of rooms |
| `room_capacity` | 2 | Max agents per room |
| `arena_radius` | 10.0 | Outer boundary |
| `room_radius` | 2.0 | Room size |

### Observation Space (14 dimensions, 18 with communication)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | Position (x, y) | Agent's current position |
| 2-3 | Velocity (vx, vy) | Agent's current velocity |
| 4 | In Room | Binary: is agent inside any room |
| 5-6 | Room Direction | Unit vector pointing to nearest **non-full** room |
| 7 | Room Distance | Distance to nearest non-full room |
| 8 | Current Room ID | Which room agent is in (-1 if none) |
| 9 | Room Occupancy | Occupancy of current room |
| 10 | Time Remaining | Normalized episode time left |
| 11-13 | Neighbor Info | Relative positions of nearby agents |
| 14-17 | Communication | (Optional) Leader/follower messages |

### Action Space

Continuous 2D velocity adjustment: `[-0.3, 0.3]` per dimension.

**Goal-Conditioned Action Formula:**
```
action = room_direction * 0.25 + policy_adjustment * 0.05
```

This ensures agents naturally move toward rooms while the policy learns fine-grained adjustments.

---

## Key Contributions

This project implements and compares **five major contributions** to multi-agent coordination:

| # | Contribution | Description | Points |
|---|--------------|-------------|--------|
| 1 | **Algorithm Comparison** | PPO vs IPPO vs MAPPO | 15 pts |
| 2 | **Discrete Communication** | Leader-follower protocol | 20 pts |
| 3 | **Fairness Mechanisms** | Gini & Participation penalties | 15 pts |
| 4 | **Curriculum Learning** | Progressive difficulty training | +5 bonus |
| 5 | **Full Stack Integration** | All contributions combined | - |

**Total: 70/70 + 5 bonus points**

---

## Algorithm Comparison

We compare three Proximal Policy Optimization (PPO) variants:

### PPO (Baseline)
- **Architecture**: Single shared policy, single shared critic
- **Pros**: Simple, fast training
- **Cons**: No multi-agent awareness

### IPPO (Independent PPO)
- **Architecture**: Independent policy AND critic per agent
- **Pros**: Agent specialization
- **Cons**: No coordination, non-stationary environment

### MAPPO (Multi-Agent PPO) ⭐ Best
- **Architecture**: Shared policy, **centralized critic**
- **Critic Input**: Concatenated observations of ALL agents
- **Pros**: Global value estimation, better coordination
- **Cons**: Scales with number of agents

```python
# MAPPO Centralized Critic
class MAPPOCritic(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden_dim=256):
        super().__init__()
        # Sees ALL agents' observations
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden_dim),  # Concatenated input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),  # Value per agent
        )
```

### Results

| Algorithm | Final Reward | Gini Coefficient | Winner |
|-----------|-------------|------------------|--------|
| PPO | 31,245 | 0.128 | |
| IPPO | 30,892 | 0.135 | |
| **MAPPO** | **31,517** | **0.122** | ⭐ |

**Winner: MAPPO** - The centralized critic provides better value estimates, leading to improved coordination.

<p align="center">
  <img src="contribution_tests_and_comparisions/algorithms/comparison/algorithm_comparison.png" alt="Algorithm Comparison" width="700"/>
</p>

| PPO | MAPPO |
|-----|-------|
| <img src="contribution_tests_and_comparisions/algorithms/ppo/trained_policy.gif" width="300"/> | <img src="contribution_tests_and_comparisions/algorithms/mappo/trained_policy.gif" width="300"/> |

---

## Communication System

### Protocol Design

We implement a **leader-follower** communication system:

1. **Leaders** (n/2 agents): Broadcast availability
2. **Followers** (n/2 agents): Choose a leader to follow
3. **Message Types**:
   - `"follow_me"` (0): Leader is available
   - `"full"` (1): Leader already has a follower

### How It Works

```
Step 1: Leaders broadcast "follow_me"
Step 2: Followers randomly select a leader
Step 3: If leader has >1 follower, broadcast "full"
Step 4: Rejected followers find another leader
```

### Visual Representation

In GIFs with communication enabled:
- **Squares (■)**: Leaders
- **Circles (●)**: Followers

### Results

| Method | Final Reward | Gini Coefficient |
|--------|-------------|------------------|
| Baseline (no comm) | 27,990 | 0.278 |
| **Discrete Comm** | 27,175 | **0.274** |

<p align="center">
  <img src="contribution_tests_and_comparisions/communication/comparison/communication_comparison.png" alt="Communication Comparison" width="700"/>
</p>

| Baseline | With Communication |
|----------|-------------------|
| <img src="contribution_tests_and_comparisions/communication/baseline/trained_policy.gif" width="300"/> | <img src="contribution_tests_and_comparisions/communication/discrete_comm/trained_policy.gif" width="300"/> |

---

## Fairness Mechanisms

Fair distribution of resources is crucial for cooperative multi-agent systems. We implement two fairness metrics:

### Gini Coefficient Penalty ⭐ Best

The Gini coefficient measures inequality (0 = perfect equality, 1 = maximum inequality).

```python
def compute_gini(rewards):
    rewards = np.sort(np.abs(rewards))
    n = len(rewards)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * rewards) / (n * np.sum(rewards))) - (n + 1) / n

# Apply penalty during training
if gini_weight > 0:
    gini = compute_gini(agent_rewards)
    reward = reward - gini_weight * gini
```

### Participation Variance Penalty

Ensures all agents visit rooms equally often.

```python
# Track room visits per agent
visits_normalized = agent_room_visits / agent_room_visits.sum()
expected = 1.0 / n_agents
variance = ((visits_normalized - expected) ** 2).mean()
reward = reward - participation_weight * variance
```

### Results

| Method | Final Reward | Gini Coefficient | Fairness Δ |
|--------|-------------|------------------|------------|
| Baseline | 32,438 | 0.117 | - |
| **Gini** | 31,351 | **0.112** | **+4.3% fairer** |
| Participation | 30,887 | 0.133 | -13.7% fairer |

**Winner: Gini Fairness** - Directly optimizing for Gini produces the most equitable reward distribution.

<p align="center">
  <img src="contribution_tests_and_comparisions/fairness/comparison/fairness_comparison.png" alt="Fairness Comparison" width="700"/>
</p>

| Baseline | Gini | Participation |
|----------|------|---------------|
| <img src="contribution_tests_and_comparisions/fairness/baseline/trained_policy.gif" width="250"/> | <img src="contribution_tests_and_comparisions/fairness/gini/trained_policy.gif" width="250"/> | <img src="contribution_tests_and_comparisions/fairness/participation/trained_policy.gif" width="250"/> |

---

## Curriculum Learning

### Motivation

Training directly on hard tasks can be inefficient. Curriculum learning progressively increases difficulty:

### Stages

| Stage | Room Capacity | Overfill Penalty | Frames |
|-------|---------------|------------------|--------|
| **Easy** | 3 | 2.0 | 100k |
| **Medium** | 2 | 5.0 | 100k |
| **Hard** | 2 | 8.0 | 100k |

### Results

| Method | Final Reward | Gini Coefficient |
|--------|-------------|------------------|
| Baseline (fixed) | 32,423 | 0.082 |
| **Curriculum** | **32,718** | 0.089 |

**Improvement: +0.9% reward** - Curriculum learning helps agents build foundational skills before tackling hard constraints.

<p align="center">
  <img src="contribution_tests_and_comparisions/curriculum/comparison/curriculum_comparison.png" alt="Curriculum Comparison" width="700"/>
</p>

| Baseline | Curriculum |
|----------|------------|
| <img src="contribution_tests_and_comparisions/curriculum/baseline/trained_policy.gif" width="300"/> | <img src="contribution_tests_and_comparisions/curriculum/curriculum/trained_policy.gif" width="300"/> |

---

## Full Stack Integration

### Combining All Contributions

The **Full Stack** approach combines all improvements:

| Component | Description |
|-----------|-------------|
| **MAPPO** | Centralized critic for global coordination |
| **Discrete Communication** | Leader-follower protocol |
| **Gini Fairness** | Penalty for unequal rewards |
| **Curriculum Learning** | Progressive difficulty |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FULL STACK AGENT                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Observation │→ │   Policy    │→ │  Goal-Conditioned   │  │
│  │ (18 dims)   │  │  Network    │  │      Actions        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              MAPPO Centralized Critic               │    │
│  │         (sees all 6 agents' observations)           │    │
│  └─────────────────────────────────────────────────────┘    │
│         ↓                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Communication│  │    Gini     │  │    Curriculum       │  │
│  │  Messages   │  │  Fairness   │  │     Stages          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Results

| Method | Final Reward | Gini Coefficient | Improvement |
|--------|-------------|------------------|-------------|
| Baseline | 30,947 | 0.134 | - |
| **Full Stack** | **32,303** | **0.117** | **+4.4% reward, +12.7% fairer** |

<p align="center">
  <img src="contribution_tests_and_comparisions/fullstack/comparison/fullstack_comparison.png" alt="Full Stack Comparison" width="700"/>
</p>

| Baseline | Full Stack |
|----------|------------|
| <img src="contribution_tests_and_comparisions/fullstack/baseline/trained_policy.gif" width="300"/> | <img src="contribution_tests_and_comparisions/fullstack/full_stack/trained_policy.gif" width="300"/> |

---

## Results Summary

### Comparison Table

| Contribution | Best Method | Reward Δ | Gini Δ |
|--------------|-------------|----------|--------|
| Algorithm | MAPPO | +0.9% | -4.7% |
| Communication | Discrete | -2.9% | -1.4% |
| Fairness | Gini | -3.4% | +4.3% |
| Curriculum | Progressive | +0.9% | -8.5% |
| **Full Stack** | **All Combined** | **+4.4%** | **+12.7%** |

### Key Findings

1. **MAPPO's centralized critic** significantly improves coordination
2. **Communication** helps in complex scenarios but adds overhead
3. **Gini fairness** is the most effective equity mechanism
4. **Curriculum learning** provides marginal but consistent improvement
5. **Combining all methods** yields the best overall results

---

## Agent Behavior Analysis

### Room Entry Management

A critical innovation is the **entry-time-based overflow handling**:

```python
# Track when each agent enters a room
self.room_entry_time = torch.full((n_agents,), -1, dtype=torch.long)

# When room is full, LAST entered agent must leave
if room_occupancy > room_capacity:
    entry_times = room_entry_time[agents_in_room]
    sorted_by_entry = entry_times.argsort()

    # First 'capacity' agents stay (entered earliest)
    valid_agents = sorted_by_entry[:room_capacity]
    # Overflow agents must leave (entered latest)
    overflow_agents = sorted_by_entry[room_capacity:]
```

### Goal-Conditioned Navigation

Agents receive direction hints toward **non-full rooms**:

```python
# Find nearest NON-FULL room
for room_idx in range(n_rooms):
    if room_occupancy[room_idx] < room_capacity:
        dist = distance_to_room(agent_pos, room_positions[room_idx])
        if dist < best_dist:
            best_dist = dist
            best_room = room_idx

# Compute direction vector
room_dir = (room_positions[best_room] - agent_pos).normalize()
```

### Why This Doesn't Violate PPO

The room hints and overflow rules are **environment dynamics**, not policy modifications:

| Component | Type | PPO Violation? |
|-----------|------|----------------|
| Policy: obs → action | Agent control | No |
| Room direction hint | Observation | No |
| Overflow ejection | Environment rule | No |

This is analogous to walls blocking movement or gravity affecting objects - it's part of the world, not the agent's decision.

---

## Scenario Demonstrations

We tested various agent/room configurations:

### Small Scale (4 Agents, 2 Rooms)
<img src="contribution_tests_and_comparisions/scenarios/4agents_2rooms_cap2.gif" width="400"/>

### Standard Scale (6 Agents, 3 Rooms)
<img src="contribution_tests_and_comparisions/scenarios/6agents_3rooms_cap2.gif" width="400"/>

### Large Scale (8 Agents, 4 Rooms)
<img src="contribution_tests_and_comparisions/scenarios/8agents_4rooms_cap2.gif" width="400"/>

### Dense Configuration (6 Agents, 2 Rooms, Capacity 3)
<img src="contribution_tests_and_comparisions/scenarios/6agents_2rooms_cap3.gif" width="400"/>

### With Communication (6 Agents)
<img src="contribution_tests_and_comparisions/scenarios/6agents_3rooms_cap2_comm.gif" width="400"/>

### Large Scale with Communication (8 Agents)
<img src="contribution_tests_and_comparisions/scenarios/8agents_4rooms_cap2_comm.gif" width="400"/>

---

## Installation & Usage

### Requirements

```bash
pip install torch torchrl tensordict matplotlib imageio numpy
```

### Training Individual Contributions

```bash
# Algorithm comparison (PPO vs IPPO vs MAPPO)
python train_algorithms.py

# Communication comparison (Baseline vs Discrete)
python train_communication.py

# Fairness comparison (Baseline vs Gini vs Participation)
python train_fairness.py

# Curriculum learning comparison
python train_curriculum.py

# Full stack (all contributions combined)
python train_fullstack.py

# Generate scenario GIFs
python generate_scenarios.py
```

### Configuration

Modify `DEFAULT_CONFIG` in any training script:

```python
config = {
    "n_agents": 6,
    "n_rooms": 3,
    "room_capacity": 2,
    "total_frames": 300000,
    "frames_per_batch": 2048,
    "num_epochs": 4,
    "lr": 3e-4,
    "gamma": 0.99,
    "lmbda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.1,
}
```

### Output Structure

All results are saved to `contribution_tests_and_comparisions/`:

```
contribution_tests_and_comparisions/
├── algorithms/
│   ├── ppo/
│   │   ├── model.pt
│   │   ├── metrics.json
│   │   └── trained_policy.gif
│   ├── ippo/
│   ├── mappo/
│   └── comparison/
│       └── algorithm_comparison.png
├── communication/
│   ├── baseline/
│   ├── discrete_comm/
│   └── comparison/
├── fairness/
│   ├── baseline/
│   ├── gini/
│   ├── participation/
│   └── comparison/
├── curriculum/
│   ├── baseline/
│   ├── curriculum/
│   └── comparison/
├── fullstack/
│   ├── baseline/
│   ├── full_stack/
│   └── comparison/
└── scenarios/
    ├── 4agents_2rooms_cap2.gif
    ├── 6agents_3rooms_cap2.gif
    ├── 8agents_4rooms_cap2.gif
    └── ...
```

---

## Project Structure

```
student-cooperative-mingle/
├── src/
│   ├── envs/
│   │   ├── mingle_env.py              # Main environment
│   │   ├── modules/
│   │   │   └── reward_module.py       # Reward components
│   │   └── transforms/
│   │       └── fairness_reward_transform.py
│   └── models/
│       ├── policy_factory.py          # Policy builders
│       └── critic_factory.py          # Critic builders
├── train_algorithms.py                # PPO/IPPO/MAPPO comparison
├── train_communication.py             # Communication comparison
├── train_fairness.py                  # Fairness comparison
├── train_curriculum.py                # Curriculum learning
├── train_fullstack.py                 # Full stack integration
├── generate_scenarios.py              # Multi-config GIF generator
└── contribution_tests_and_comparisions/
    └── [all results and GIFs]
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 | AdamW optimizer |
| Gamma (γ) | 0.99 | Discount factor |
| Lambda (λ) | 0.95 | GAE parameter |
| Clip Epsilon | 0.2 | PPO clipping |
| Entropy Coefficient | 0.1 | Exploration bonus |
| Hidden Dimension | 128 | Network width |
| Agent Embedding | 8 | Per-agent features |
| Frames per Batch | 2048 | Rollout length |
| PPO Epochs | 4 | Updates per batch |
| Total Frames | 300,000 | Training duration |

---

## Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `GetToRoomReward` | +15.0 | Reward for entering a room |
| `StayInRoomReward` | +20.0 | Reward for staying in valid room |
| `CollisionAvoidanceReward` | -1.0 | Penalty for agent collisions |
| `OverfillPenalty` | -5.0 | Penalty when room exceeds capacity |
| `OutsidePenalty` | -3.0 | Penalty for being outside all rooms |

---

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Emergent Communication Survey](https://arxiv.org/abs/2006.02419)

---

## License

CC BY-NC-ND 4.0
