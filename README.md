# Cooperative Mingle: Multi-Agent Reinforcement Learning for Room Allocation

A comprehensive Multi-Agent Reinforcement Learning (MARL) framework for solving the cooperative room allocation problem. Agents must coordinate to distribute themselves fairly across rooms while respecting capacity constraints.

**Course:** Collective Intelligence - Multi-Agent Reinforcement Learning
**Semester:** 2025/26/2
**Assignment 2:** Cooperative Mingle

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train with Hydra
python train_hydra.py algorithm=mappo

# Run all experiments as sweep
python train_hydra.py --multirun algorithm=ppo,ippo,mappo

# Run sweeps in parallel
python train_hydra.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 algorithm=ppo,ippo,mappo

# Docker
docker build -t cooperative-mingle .
docker run cooperative-mingle  # runs tests
docker run cooperative-mingle python train_hydra.py  # training
```

---

## Semester Contribution: Task 3 — Multi-Objective Coordination

**Research question.** Can we map the Pareto frontier between coordination efficiency and fairness in the Cooperative Mingle environment, and do different reward weights lead to different coordination behavior?

**Hypothesis.** We expected that higher efficiency weight would increase total reward and room occupancy, while lower efficiency weight would keep agent participation more balanced. By changing the scalarization parameter `alpha`, we wanted to observe how the policy moves between efficiency-focused and fairness-focused behavior.

**Implementation summary.** For Task 3, we implemented a multi-objective reward setup using scalarization:

```
R = alpha * R_efficiency + (1 - alpha) * R_fairness
```

The implementation includes three main reward components: EfficiencyReward, FairnessReward, and MultiObjectiveReward. The efficiency component rewards successful room claiming and occupancy, while the fairness component supports multiple fairness metrics, including Gini-based fairness, Jain fairness, participation variance, exclusion variance, and participation range.

We also added Pareto-specific tests in `tests/test_pareto.py`. These tests check the participation range metric, Jain fairness behavior, scalarization correctness, output shape, and the effect of changing alpha. The final Task 3 tests passed successfully.

### Experiment setup

We evaluated the following alpha values: `0.0, 0.25, 0.5, 0.75, 1.0`

The experiments were run on both small and large scenario regimes.

**Small scenarios:**
```
4 agents / 2 rooms
4 agents / 3 rooms
6 agents / 2 rooms
6 agents / 3 rooms
```

**Large scenarios:**
```
8 agents / 4 rooms
8 agents / 5 rooms
10 agents / 4 rooms
10 agents / 5 rooms
```

Most conditions were evaluated with 5 fixed seeds. The 8-agent / 4-room large condition also includes one additional smoke-validation run, so it has 6 seeds in the aggregated results.

Reproduce the artifacts with the Pareto commands under [Installation & Usage](#installation--usage); the original run outputs (CSVs, plots, GIFs) live on the `2025/26/2` semester branch.

### Key results
Across both small and large scenarios, increasing alpha generally increased the mean reward. This matches the expected behavior, since higher alpha values give more weight to the efficiency objective.

For example, in the large 10-agent / 5-room scenario, the mean reward increased from approximately 706.07 at alpha = 0.0 to 781.45 at alpha = 1.0.

The Gini coefficient remained relatively low overall, but it increased as alpha grew. This suggests that stronger efficiency weighting improved reward while slightly reducing measured fairness.

### Conclusions and limitations

The final implementation supports multi-objective reward composition, alpha-based scalarization, Hydra sweeps, fixed-seed evaluation, aggregated CSV reporting, GIF generation, and Pareto visualization.

The main limitation is that the training budget was kept short so that the full small and large sweeps could run on CPU. Because of this, the results should be interpreted as exploratory Pareto analysis rather than fully converged MARL policies. Another limitation is that the fairness metrics showed limited variation across alpha values. Future experiments should use longer training and include additional behavioral statistics such as room switches, forced exits, and collision rates.

### Future work

Future work could include longer training runs, stronger fairness objectives, constrained multi-objective optimization, and more detailed behavior analysis across the Pareto frontier. Additional plots for room switching, exclusion events, and collision rates would make the policy comparison more informative.

### Individual contributions

Both team members worked on the Task 3 implementation. We first developed separate versions, compared the approaches, and then merged the useful parts into the final Pareto pipeline.
- Patrik: implemented the main Pareto reward structure, including efficiency and fairness reward components, prepared Hydra Pareto sweeps, ran the small-scenario experiments, generated aggregated CSV results, produced Pareto visualizations, and will deliver the oral presentation.
- Anna: validated the merged Task 3 pipeline, ran and checked the final tests and smoke training, executed the full large-scenario sweep, organized the Pareto artifacts, created the combined small + large aggregated CSV, updated the Pareto artifact README and the root README, and prepared the presentation slides.
---

## Table of Contents

- [Quick Start](#quick-start)
- [Semester Contribution: Task 3 — Multi-Objective Coordination](#semester-contribution-task-3--multi-objective-coordination)
- [Problem Description](#problem-description)
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

---

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- TorchRL
- Hydra-core
- Matplotlib
- ImageIO
- Pytest

### Training

All experiments are driven through `train_hydra.py` with Hydra config overrides:

```bash
# Single run (override algorithm / fairness / communication as needed)
python train_hydra.py algorithm=mappo fairness=gini communication=discrete

# Sweeps (multiple experiments)
python train_hydra.py --multirun algorithm=ppo,ippo,mappo
python train_hydra.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 algorithm=ppo,ippo,mappo
```

### Pareto analysis (Task 3)

```bash
# Multi-objective alpha sweep (small / large), then aggregate + plot the frontier
python train_hydra.py --multirun +sweep=pareto
python train_hydra.py --multirun +sweep=pareto_large
python analyze_pareto.py
```

### Docker

```bash
# Build
docker build -t cooperative-mingle .

# Run tests
docker run cooperative-mingle

# Training with GPU
docker run --gpus all cooperative-mingle python train_hydra.py
```

---

## Project Structure

```
student-cooperative-mingle/
├── src/
│   ├── envs/
│   │   ├── mingle_env.py              # Main environment
│   │   └── modules/
│   │       └── reward_module.py       # Reward components
│   └── models/
│       ├── policy_factory.py          # Policy builders
│       └── critic_factory.py          # Critic builders
├── configs/                           # Hydra configs
│   ├── config.yaml
│   ├── algorithm/
│   ├── communication/
│   ├── fairness/
│   ├── curriculum/
│   └── sweep/
├── tests/                             # Unit/smoke tests
│   ├── test_environment.py
│   ├── test_communication.py
│   └── test_fairness.py
├── train_hydra.py                     # Hydra training entry point
├── analyze_pareto.py                  # Pareto sweep aggregation + plots (Task 3)
├── Dockerfile                         # Docker support
└── requirements.txt
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


