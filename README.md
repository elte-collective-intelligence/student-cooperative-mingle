# Cooperative Mingle


## Table of Contents
1. Base Environment
2. Dynamic Environment
3. Reward Modules
4. Metrics System
5. Components Builder
6. Training Pipeline
7. Evaluation Pipeline
8. Testing Framework
9. GIF Generation
10. Generalized Advantage Estimation (GAE)
11. Configuration Files

## MingleEnv Environment

`MingleEnv` simulates an environment where agents move inside a circular arena, alternating between spinning and claiming phases, and interacting with rooms distributed around the arena.

### Initialization Parameters

| Parameter             | Type                     | Description                                                                                   |
|-----------------------|--------------------------|-----------------------------------------------------------------------------------------------|
| `n_agents`            | `int`                    | Number of agents in the environment                                                          |
| `n_rooms`             | `int`                    | Number of rooms distributed around the arena                                                 |
| `arena_radius`        | `float`                  | Outer radius of the arena                                                                    |
| `center_radius`       | `float`                  | Radius of the central region where agents spin                                               |
| `max_steps`           | `int`                    | Maximum number of steps in an episode                                                        |
| `spinning_phase_range`| `Tuple[int, int]`        | Range (min, max) for random spinning phase duration                                          |
| `room_radius`         | `float`                  | Radius of each room (used for positioning)                                                  |
| `room_capacity`       | `int`                    | Maximum number of agents that can occupy a room                                             |
| `reward_modules`      | `Optional[List[RewardModule]]` | List of callable modules to provide rewards                                              |
| `reward_managers`     | `Optional[dict]`         | Dictionary of reward managers (optional)                                                     |
| `phase_mode`          | `str`                    | Phase behavior control mode ("both", "claiming", etc.)                                      |

---

### Key Methods

- `__init__(...)`  
  Initialize the environment with specified parameters.

- `_init_agents()`  
  Randomly initialize agent positions inside the center circle.

- `_init_rooms()`  
  Initialize rooms evenly spaced around the arena circumference.

- `_rotate_positions(angle_rad: float)`  
  Rotate all agent positions around the arena center by `angle_rad` radians.

- `_enforce_boundaries()`  
  Clamp agent positions to remain within the arena radius.

- `_compute_observations() -> TensorDict`  
  Compute a 14-dimensional observation vector per agent including:
  - Distance and direction to center
  - Distance and direction to closest room
  - Distance and direction to nearest agent
  - Room occupancy info
  - Current phase flag

- `_compute_rewards() -> Tensor`  
  Calculate rewards by aggregating outputs from registered reward modules.

- `_reset(tensordict: Optional[TensorDict] = None) -> TensorDict`  
  Reset the environment state for a new episode.

- `_step(tensordict: TensorDict) -> TensorDict`  
  Advance the simulation by one step, applying actions, updating positions, phases, observations, rewards, and done flags.

- `_set_seed(seed: Optional[int]) -> None`  
  Set the random seed for reproducibility.

---

### Observation Vector (Per Agent)

The observation vector has 14 features:

| Index | Feature Description                                           |
|-------|--------------------------------------------------------------|
| 0     | Normalized distance to center (`dist_center / arena_radius`) |
| 1     | Direction to center (x component)                            |
| 2     | Direction to center (y component)                            |
| 3     | Signed distance to center edge (`dist_center - center_radius`)|
| 4     | Normalized distance to closest room center                   |
| 5     | Direction to closest room (x component)                      |
| 6     | Direction to closest room (y component)                      |
| 7     | Signed distance to closest room edge                          |
| 8     | Room capacity (constant)                                     |
| 9     | Current room occupancy ratio                                  |
| 10    | Normalized distance to nearest agent                          |
| 11    | Direction to nearest agent (x component)                     |
| 12    | Direction to nearest agent (y component)                     |
| 13    | Current phase flag (1 if spinning, 0 if claiming)            |

---

### Example Usage

```python
from src.envs.mingle_env import MingleEnv
import torch

env = MingleEnv(
    n_agents=4,
    n_rooms=3,
    arena_radius=10.0,
    center_radius=3.0,
    max_steps=300,
    spinning_phase_range=(50, 100),
    room_radius=3.0,
    room_capacity=2,
    phase_mode="both"
)

tensordict = env.reset()
for _ in range(env.max_steps):
    # Example random actions: small random movement vectors for each agent
    actions = torch.randn(env.n_agents, 2) * 0.1
    tensordict = env.step({"action": actions})
```

## DynamicMingleEnv

`DynamicMingleEnv` simulates a multi-agent environment where agents move in a circular arena with two main phases: **spinning** and **claiming**. Agents interact with rooms placed evenly around the arena's perimeter while navigating and competing for occupancy.

---

### Overview

- **Agents:** Move within a circular arena and interact with rooms.
- **Phases:**
  - **Spinning:** Agents move within the center radius and rotate collectively.
  - **Claiming:** Agents leave the center to claim rooms distributed around the arena.
- **Rooms:** Positioned evenly on the arena edge; each room has a capacity limit.

---

### Key Features

- **Configurable parameters:** Number of agents, rooms, arena radius, max steps, phase duration, room capacity, max speed, etc.
- **Agent dynamics:** Includes velocity, facing direction, momentum, and boundary enforcement.
- **Phase control:** Automatically transitions from spinning to claiming based on steps or mode.
- **Observation space:** Includes detailed information on agent position, velocity, room distances, occupancy, facing direction, and phase.
- **Reward integration:** Supports flexible reward modules and managers per phase.
- **Environment interface:** Compatible with `tensordict` for observations, actions, rewards, and termination flags.

---

### Constructor Parameters

| Parameter          | Type                 | Default     | Description                                                |
|--------------------|----------------------|-------------|------------------------------------------------------------|
| `n_agents`         | `int`                | `2`         | Number of agents in the environment.                      |
| `n_rooms`          | `int`                | `2`         | Number of rooms placed around the arena.                  |
| `arena_radius`     | `float`              | `10.0`      | Radius of the circular arena.                             |
| `center_radius`    | `float`              | `3.0`       | Radius of the central spinning region.                    |
| `max_steps`        | `int`                | `300`       | Maximum number of steps per episode.                       |
| `spinning_phase_range` | `Tuple[int, int]` | `(50, 100)` | Random range for spinning phase duration.                 |
| `room_radius`      | `float`              | `3.0`       | Radius of each room.                                       |
| `room_capacity`    | `int`                | `2`         | Maximum number of agents allowed per room.                 |
| `max_speed`        | `float`              | `0.5`       | Maximum speed for agents.                                  |
| `reward_modules`   | `List[RewardModule]` or `None` | `None` | Reward modules for computing rewards.                      |
| `reward_managers`  | `dict` or `None`     | `None`      | Reward manager instances per phase.                        |
| `phase_mode`       | `str`                | `"both"`    | Controls phase behavior ("both", "claiming", etc.).       |

---

### Core Methods

`_init_agents()`

Randomly initializes agent positions within the center circle.

`_init_rooms()`

Positions rooms evenly spaced on the arena edge with some margin.

`_step(tensordict)`

Performs one environment step:
- Updates agent facing and velocity based on actions.
- Moves agents and enforces arena boundaries.
- Computes room assignments and occupancy.
- Handles phase transitions.
- Returns updated observations, rewards, done flags, and termination status.

`_compute_observations()`

Builds a rich observation tensor per agent containing:
- Distances and relative angles to center, closest agent, and closest room.
- Room occupancy and assignment.
- Velocity and facing direction info.
- Current phase encoded as a tensor.

`_compute_rewards()`

Aggregates rewards from active reward modules or reward managers based on the current phase.

`_reset(tensordict)`

Resets environment state, agent and room positions, phase, and velocities.

---

### Usage Example

```python
env = DynamicMingleEnv(n_agents=4, n_rooms=3)
tensordict = env.reset()

for _ in range(env.max_steps):
    actions = get_agent_actions_somehow()
    tensordict = env.step(tensordict.set("action", actions))
    obs = tensordict["observation"]
    reward = tensordict["reward"]
    done = tensordict["done"]
    if done.any():
        break
```

## Reward Modules and Reward Manager

This module defines a set of reward modules used in the `MingleEnv` environment to shape agent behavior during different phases ("spinning" or "claiming"). It also includes a `RewardManager` to handle activation of rewards based on performance thresholds.

---

### Abstract Base Class: `RewardModule`

- **Purpose:** Base class for all reward modules.
- **Phase Mode:** Can be `"spinning"`, `"claiming"`, or `"both"` to control when the reward applies.
- **Activation:** Each module can be activated or deactivated.
- **Usage:** Subclasses implement `_reward(env)` to compute rewards given an environment state.

### Key Methods

- `__call__(env: MingleEnv) -> Tensor`  
  Returns zero reward if the module's phase doesn't match the env's current phase or if inactive; otherwise calls `_reward`.

- `_activate()` / `_deactivate()`  
  Enable or disable the module.

- `_reward(env: MingleEnv) -> Tensor`  
  Must be implemented by subclasses; computes the reward tensor for all agents.

---

## Concrete Reward Modules

### 1. `CloseToCenterReward`

- Rewards agents closer to the center during the applicable phase.
- Reward = 1 - (distance to center / center radius), clipped between 0 and 1.

---

### 2. `InsideCenterReward`

- Rewards agents inside the center region.
- Parameters:  
  - `inside_reward`: positive reward for agents inside center (default 1.0)  
  - `outside_penalty`: negative penalty for agents outside center (default -1.0)

---

### 3. `CollisionAvoidanceReward`

- Penalizes agents that are closer than a minimum distance to any other agent.
- Parameters:  
  - `min_distance`: minimum safe distance (default 0.5)  
  - `penalty`: penalty applied when too close (default 1.0)

---

### 4. `GetToRoomReward`

- Rewards agents for approaching rooms that are not full.
- Penalizes agents approaching full rooms.
- Uses quadratic scaling based on distance to closest room.
- Parameters:  
  - `max_reward`: maximum positive reward (default 1.0)  
  - `min_distance`: minimum distance for scaling (default 0.0)  
  - `full_room_penalty`: penalty for approaching full rooms (default 1.0)

---

### 5. `StayInRoomReward`

- Rewards agents for staying inside rooms.
- Penalizes agents outside rooms and those in overfilled rooms.
- Parameters:  
  - `max_reward`: maximum reward (default 1.0)  
  - `outside_penalty`: penalty for being outside any room (default -1.0)  
  - `overfill_penalty`: penalty for being in an overfilled room (default -1.0)

---

## Reward Manager

This section describes the interactive selection and threshold-based activation system for reward modules used during training in the MingleEnv. It also explains the RewardManager class that dynamically activates reward modules based on agent performance during different environment phases.
I
### Interactive Reward Module Selector: select_reward_modules

The function select_reward_modules() allows users to interactively choose which reward modules to include for each phase ("spinning" and "claiming"). For each phase:

- A list of available reward modules is shown.

- The user selects modules by index.

- For each selected module (after the first), the user defines a mean reward threshold.

- Modules are activated sequentially during training when the mean reward exceeds the threshold for the previous module.

Example usage:
```python
selected, thresholds = select_reward_modules()
# selected = {
#   "spinning": [InsideCenterReward(...), CollisionAvoidanceReward(...), ...],
#   "claiming": [...]
# }
# thresholds = {
#   "spinning": [0.2, 0.5, ...],
#   "claiming": [...]
# }
```
### RewardManager Class

The RewardManager handles sequential activation of reward modules within a given phase, based on the mean reward achieved by agents.

Features:

- Maintains a list of reward modules and associated activation thresholds.

- Starts with the first module active.

- When the current mean reward exceeds the threshold, activates the next module.

- Computes the combined reward from all active modules.

Key methods:

- update(reward_mean: float): Checks if the current mean reward passes the threshold to activate the next module.

- __call__(env) -> Tensor: Returns the aggregated reward tensor by summing outputs from all active reward modules for the given environment state.

Usage example:
```python
reward_manager = RewardManager(selected["spinning"], thresholds["spinning"], phase="spinning")

 During training loop:
mean_reward = compute_mean_reward()  # Your logic here
reward_manager.update(mean_reward)

# Get combined rewards for all agents
rewards = reward_manager(env)
```

### Integration Notes

- This design supports phase-specific reward shaping by allowing different reward modules to be activated as agent performance improves.

- The threshold-based sequential activation encourages progressive learning by introducing more complex objectives as simpler ones are mastered.

- The interactive selection helps customize training objectives without code changes, improving experimentation flexibility.

## Metrics System

The metrics system provides comprehensive tracking of agent behavior and environment dynamics through the `MetricModule` base class and its implementations.

### Base Class: MetricModule

All metrics inherit from `MetricModule`:

```python
from src.envs.modules.metric_module import MetricModule

class CustomMetric(MetricModule):
    def __init__(self):
        super().__init__("custom_metric_name")

    def reset(self):
        # Reset internal state
        pass

    def update(self, env):
        # Update metric with current environment state
        pass

    def compute(self):
        # Return final metric values as dict
        return {self.name: value}
```

### Available Metrics

#### 1. `CollisionRateMetric`
Tracks the rate of agent collisions based on minimum distance threshold.

#### 2. `RoomOccupancyRateMetric`
Measures the percentage of agents currently inside rooms.

#### 3. `CenterPresenceMetric`
Tracks the proportion of time agents spend in the center area.

#### 4. `AverageStepDistanceMetric`
Calculates the average distance agents move per step.

#### 5. `IdleAgentRateMetric`
Identifies agents that are not moving (below threshold).

#### 6. `RoomSwitchesMetric`
Counts how often agents switch between rooms.

#### 7. `PhaseTimeMetric`
Tracks time distribution across different environment phases.

#### 8. `AgentDensityMetric`
Measures local agent density within a specified radius.

#### 9. `MaxDistanceFromCenterMetric`
Records the maximum distance any agent reaches from center.

#### 10. `MinAgentDistanceMetric`
Tracks the minimum distance between any two agents.

#### 11. `AverageRoomDistanceMetric`
Calculates average distance from agents to nearest room.

#### 12. `AgentMovementVarianceMetric`
Measures variance in agent movement patterns.

### Usage in Training

Metrics are automatically integrated in the training pipeline through pipeline.py:

```python
# Metrics are used during evaluation
from src.eval.pipeline import evaluate

# Run evaluation with all metrics
evaluate(policy_module, env, device, num_episodes=100)
```

## Policy and Critic Builders

This section covers two crucial functions for constructing the core neural network models used in the training pipeline: the policy (actor) and the critic (value function). Both are designed for multi-agent environments and leverage TorchRL modules.

---

### `build_policy`

Creates a **ProbabilisticActor** policy model that outputs a distribution over actions conditioned on observations. This allows the agent to act stochastically, essential for algorithms like PPO.

**Key Features:**
- Uses a multi-agent MLP (`MultiAgentMLP`) to process observations, with configurable depth, width, and activation function.
- Outputs mean (`loc`) and scale (`scale`) parameters for a Normal distribution, constrained to a minimum scale using a custom module.
- Wraps outputs in a `TanhNormal` distribution to bound actions within environment limits.
- Supports centralized or decentralized architectures and parameter sharing across agents.
- Converts raw network outputs to action distribution parameters suitable for sampling.

**Arguments:**
- `env`: Environment instance providing observation and action specs.
- `policy_config`: Dict defining MLP architecture, activation, minimum scale, and sharing options.
- `device`: Torch device (CPU/GPU).

**Returns:**
- A configured `ProbabilisticActor` compatible with TorchRL's RL loop.

---

### `build_critic`

Creates a **ValueOperator** serving as the critic network, estimating the value function for each agent.

**Key Features:**
- Uses `MultiAgentMLP` with customizable depth, width, and activation function.
- Produces a scalar value estimate per agent.
- Supports centralized value estimation and parameter sharing options.
- Returns a TorchRL `ValueOperator` that integrates cleanly into the RL training pipeline.

**Arguments:**
- `env`: Environment instance for input/output shapes.
- `critic_config`: Dict specifying architecture and sharing parameters.
- `device`: Torch device.

**Returns:**
- A `ValueOperator` that outputs value estimates given observations.

---

## Build Training Components

The `build_train_components` function initializes and assembles all essential components needed for the training loop based on the provided configuration and device setup. It encapsulates environment creation, model construction, data collection, advantage estimation, loss computation, and optimization setup.

### What It Does

- **Environment Setup:**  
  Creates a transformed environment with observation normalization and optional reward modules/managers integrated. Moves the environment to the specified device (CPU/GPU).

- **Policy and Critic Networks:**  
  Builds the policy network (actor) and critic network according to the configuration, enabling the agent to learn optimal behaviors and value estimation.

- **Data Collector:**  
  Uses `SyncDataCollector` from TorchRL to collect environment interactions synchronously, managing batch sizes and frame counts.

- **Replay Buffer:**  
  Initializes a replay buffer with lazy tensor storage and a sampler without replacement to store experience batches for training mini-batches.

- **Advantage Estimation Module:**  
  Uses a stable Generalized Advantage Estimator (GAE) module to compute advantage values, improving training stability and variance reduction.

- **Loss Function:**  
  Constructs a clipped PPO loss module combining actor and critic losses, with entropy regularization to encourage exploration.

- **Optimizer and Scheduler:**  
  Sets up an AdamW optimizer for the combined loss parameters and a learning rate scheduler (StepLR) to reduce the learning rate periodically during training.

### Parameters

| Parameter        | Type                    | Description                                           |
|------------------|-------------------------|-------------------------------------------------------|
| `config`         | `dict`                  | Configuration dictionary containing environment, model, training, and PPO parameters. |
| `device`         | `torch.device`          | Device to run the components on (CPU or CUDA GPU).    |
| `reward_modules` | `Optional[List[RewardModule]]` | List of reward modules to customize environment rewards (optional). |
| `reward_managers`| `Optional[dict]`        | Managers that coordinate reward modules per training phase (optional). |

### Returns

A dictionary containing all the constructed components:

- `device`: the device used (CPU/GPU)  
- `env`: the transformed environment  
- `policy`: policy network (actor)  
- `critic`: critic network  
- `collector`: synchronous data collector  
- `replay_buffer`: replay buffer instance  
- `advantage_module`: GAE advantage estimator  
- `loss_module`: PPO loss module  
- `optimizer`: AdamW optimizer  
- `scheduler`: learning rate scheduler  

---

### Usage

This function is called before starting training to prepare all components according to your configuration and hardware setup:

```python
components = build_train_components(config, device, reward_modules=my_rewards)

## Training Pipeline

This training script orchestrates the reinforcement learning pipeline for training an agent in a custom environment. It handles data collection, training, logging, evaluation, and visualization with detailed metrics tracking and GIF generation.

### Key Features

- **Data Collection & Replay Buffer:** Collects experience batches from the environment using a collector and stores them in a replay buffer for training.
- **Multi-Epoch Training:** Performs multiple training epochs per batch with mini-batching, gradient clipping, and validation for NaN or infinite losses.
- **Reward Module Flexibility:** Supports manual configuration or predefined reward modules to customize the agent's learning objectives.
- **Logging & Visualization:** Tracks metrics such as reward, losses, learning rate, and invalid batch ratios; saves logs as JSON and generates plots.
- **GIF Generation:** Periodically creates GIFs of the agent’s behavior during training, with support for character animation.
- **Evaluation:** Runs an evaluation phase post-training, logging performance over multiple episodes.
- **Configurable Parameters:** Allows easy adjustments via config files, including training frames, batch size, epochs, logging intervals, and more.

### Usage

Run the script directly as a module. It will prompt for manual reward configuration or use defaults, then start the training loop. After training, evaluation and visualization artifacts are saved in timestamped directories.

```bash
python -m src.train.pipeline
```

## Evaluation Pipeline

The evaluation system in pipeline.py provides comprehensive assessment of trained policies.

### Core Function: `evaluate`

```python
from src.eval.pipeline import evaluate

def evaluate(policy_module, env, device, num_episodes=10, max_steps=300, out_dir=None):
    """
    Evaluates a policy across multiple episodes and computes metrics.

    Args:
        policy_module: Trained policy to evaluate
        env: Environment instance
        device: Computing device (CPU/GPU)
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        out_dir: Output directory for results

    Returns:
        Dict of metrics across episodes
    """
```

### Evaluation Process

1. **Policy Setup**: Policy is set to evaluation mode
2. **Metric Initialization**: All metric modules are reset
3. **Episode Execution**: Policy runs deterministically for specified episodes
4. **Metric Collection**: All metrics are updated each step
5. **Results Aggregation**: Statistics computed across episodes
6. **Output Generation**: JSON files and plots saved

### Output Structure

```
eval_results/
└── eval_YYYYMMDD_HHMMSS/
    ├── metrics.json          # Raw metric data
    ├── collision_rate.png    # Individual metric plots
    ├── room_occupancy_rate.png
    └── ... (one plot per metric)
```

### Integration with Training

Evaluation is automatically called at the end of training in pipeline.py:

```python
# Automatic evaluation after training
if policy_module is not None:
    eval_out_dir = os.path.join("eval_results", f"eval_metrics_{timestamp}")
    evaluate(policy_module, env, device, num_episodes=eval_episodes, max_steps=env.max_steps, out_dir=eval_out_dir)
```

## Testing Framework

The testing system in test.py provides comprehensive validation of environment and metric functionality.

### Running Tests

```bash
# Run all tests
python -m pytest tests/test.py

# Run specific test
python -m pytest tests/test.py::test_initialization

# Run with verbose output
python -m pytest tests/test.py -v
```

### Test Categories

#### Environment Tests
- **Initialization**: Validates environment setup
- **Reset Functionality**: Ensures proper state reset
- **Step Updates**: Verifies position updates
- **Boundary Enforcement**: Tests spatial constraints
- **Phase Transitions**: Validates environment phases

#### Metric Tests
Each metric has dedicated tests ensuring:
- Proper initialization and reset
- Correct state updates
- Valid output ranges
- Edge case handling

Example metric test:
```python
def test_collision_rate_metric(env):
    metric = CollisionRateMetric()
    metric.reset()
    for _ in range(5):
        metric.update(env)
    result = metric.compute()
    assert "collision_rate" in result
    assert 0 <= result["collision_rate"] <= 1
```

### Test Fixtures

The `env` fixture provides a configured environment for all tests:

```python
@pytest.fixture
def env():
    return MingleEnv(
        n_agents=4,
        n_rooms=3,
        max_steps=100,
        room_radius=1.0,
        center_radius=2.0
    )
```

## GIF Generation

The GIF generation system in gif.py creates visual representations of agent behavior.

### Core Functions

#### 1. `make_gif`
Basic GIF generation with circle representations:

```python
from src.eval.gif import make_gif

make_gif(
    env=environment,
    policy_module=trained_policy,
    steps=300,
    gif_path="outputs/simulation.gif",
    fps=10,
    device=device
)
```

#### 2. `make_gif_char`
Enhanced GIF with character sprites:

```python
from src.eval.gif import make_gif_char

make_gif_char(
    env=environment,
    policy_module=trained_policy,
    steps=300,
    gif_path="outputs/character_sim.gif",
    fps=10,
    device=device,
    use_character_animation=True,
    runner_image_paths=["path/to/runner1.png", "path/to/runner2.png"],
    standing_image_paths=["path/to/stand1.png", "path/to/stand2.png"],
    image_size=50,
    speed_threshold=0.05
)
```

### Visual Elements

- **Agents**: Colored circles or character sprites
- **Rooms**: Circles with occupancy-based coloring
- **Center Area**: Highlighted circular region
- **Movement Trails**: Optional trajectory visualization
- **State Indicators**: Agent numbering and phase information

### Integration with Training

GIFs are generated automatically during training at specified intervals:

```python
# In training pipeline
if gif_interval is not None and batch_count % gif_interval == 0:
    gif_path = os.path.join(gif_dir, f"mingle_batch_{batch_count}.gif")
    make_gif(env, policy_module, gif_path=gif_path, steps=env.max_steps)
```

### Character Animation

Character sprites are loaded from the visual_utils directory:
- **Runners**: `squidgame_runner/r1.png` through `r6.png`
- **Standing**: `squidgame_stand/s1.png` and `s2.png`

## Generalized Advantage Estimation (GAE)

The GAE implementation in gae.py provides stable advantage estimation for policy gradient methods.

### StableGAE Class

```python
from src.train.modules.gae import StableGAE

class StableGAE(GAE):
    """
    Enhanced GAE implementation with improved numerical stability
    """
```

### Usage in Training

GAE is integrated into the training pipeline as the advantage module:

```python
# In training setup
advantage_module = StableGAE(
    gamma=0.99,           # Discount factor
    lmbda=0.95,          # GAE lambda parameter
    value_network=critic, # Value function network
    average_gae=True     # Normalize advantages
)

# During training
for tensordict_data in collector:
    advantage_module(tensordict_data)  # Compute advantages
    # ... continue with policy updates
```

### Key Features

- **Numerical Stability**: Enhanced handling of edge cases
- **Memory Efficiency**: Optimized tensor operations
- **Gradient Flow**: Proper gradient computation for value updates

## Configuration Files

The project uses YAML configuration files in the configs directory for modular parameter management.

### Configuration Files Overview

#### 1. train.yaml
Training hyperparameters and settings:

```yaml
train:
  total_frames: 1200          # Total training frames
  frames_per_batch: 300       # Frames per batch
  num_epochs: 10              # Optimization epochs per batch
  minibatch_size: 50          # Minibatch size
  max_grad_norm: 1.0          # Gradient clipping
  lr: 3.0e-3                  # Learning rate
  lr_step_size: 20            # LR scheduler step size
  lr_gamma: 0.5               # LR decay factor
  log_interval: 1             # Logging frequency
  metrics_save_path: "training_metrics.json"
  eval_episodes: 100          # Evaluation episodes
```

#### 2. policy.yaml
Policy network architecture:

```yaml
policy:
  min_scale: 0.1              # Minimum action scale
  depth: 4                    # Network depth
  num_cells: 128              # Hidden units per layer
  activation: tanh            # Activation function
  share_params: false         # Parameter sharing
  centralised: false          # Centralized training
```

#### 3. critic.yaml
Critic network configuration:

```yaml
critic:
  depth: 4                    # Network depth
  num_cells: 128              # Hidden units per layer
  activation: tanh            # Activation function
  share_params: false         # Parameter sharing
  centralised: false          # Centralized training
```

#### 4. env.yaml
Environment parameters (structure inferred):

```yaml
env:
  n_agents: 8                 # Number of agents
  n_rooms: 4                  # Number of rooms
  max_steps: 300              # Episode length
  room_radius: 1.0            # Room size
  center_radius: 2.0          # Center area size
  # ... other environment parameters
```

#### 5. ppo.yaml
PPO algorithm parameters (structure inferred):

```yaml
ppo:
  clip_epsilon: 0.2           # PPO clipping parameter
  entropy_weight: 0.01        # Entropy regularization
  value_loss_weight: 0.5      # Value loss coefficient
  # ... other PPO parameters
```

### Configuration Loading

Configurations are loaded and merged using the utility in config.py:

```python
from src.utils.config import load_and_merge_configs

# Load all configurations
config = load_and_merge_configs([
    "configs/train.yaml",
    "configs/policy.yaml",
    "configs/critic.yaml",
    "configs/env.yaml",
    "configs/ppo.yaml"
])

# Access configuration values
learning_rate = config["train"]["lr"]
policy_depth = config["policy"]["depth"]
```

### Usage in Training Pipeline

Configurations are automatically loaded in pipeline.py:

```python
if __name__ == "__main__":
    config = load_and_merge_configs([
        "configs/env.yaml",
        "configs/policy.yaml",
        "configs/critic.yaml",
        "configs/ppo.yaml",
        "configs/train.yaml"
    ])

    # Use config throughout training
    components = build_train_components(config)
    logs = train(**training_params_from_config)
```
