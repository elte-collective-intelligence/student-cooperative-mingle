import pytest
import torch
from torchrl.envs import check_env_specs
from tensordict import TensorDict

from src.envs.mingle_env import MingleEnv

from src.envs.modules.metric_module import (
    AgentMovementVarianceMetric, AverageRoomDistanceMetric, CollisionRateMetric, MaxDistanceFromCenterMetric, MinAgentDistanceMetric, RoomOccupancyRateMetric, CenterPresenceMetric,
    AverageStepDistanceMetric, IdleAgentRateMetric, RoomSwitchesMetric,
    PhaseTimeMetric, AgentDensityMetric
)


@pytest.fixture
def env():
    return MingleEnv(n_agents=4, n_rooms=2)

def test_initialization(env):
    assert env.n_agents == 4
    assert env.n_rooms == 2
    assert env.room_positions is None
    assert env.agent_positions.shape == (4, 2)


def test_reset_returns_tensordict(env):
    td = env.reset()
    assert "observation" in td
    assert td["observation"].shape == (4, 14)
    assert not td["done"].any()
    assert not td["terminated"].any()
    assert env.phase in {"spinning", "claiming"}


def test_reset_initializes_positions(env):
    env.reset()
    norms = env.agent_positions.norm(dim=1)
    assert torch.all(norms <= env.center_radius + 1e-5)


def test_room_positions_on_reset(env):
    env.reset()
    assert env.room_positions.shape == (env.n_rooms, 2)
    assert env.room_occupancy.shape == (env.n_rooms,)


def test_step_updates_positions(env):
    env.reset()
    before = env.agent_positions.clone()
    actions = torch.full((env.n_agents, 2), 0.1)
    td = env._step(TensorDict({"action": actions}))
    after = env.agent_positions
    assert not torch.allclose(before, after)
    assert td["observation"].shape == (env.n_agents, 14)
    assert td["reward"].shape == (env.n_agents, 1)


def test_step_clips_speed(env):
    env.reset()
    # Deliberately use an action > max_speed
    long_action = torch.full((env.n_agents, 2), 1.0)
    env._step(TensorDict({"action": long_action}))
    assert torch.all(env.agent_positions.norm(dim=1) <= env.arena_radius)


def test_phase_transition(env):
    env.phase_mode = "both"
    env.reset()
    env.spinning_duration = 2
    env.phase = "spinning"
    for _ in range(3):
        env._step(TensorDict({"action": torch.zeros((env.n_agents, 2))}))
    assert env.phase == "claiming"


def test_enforces_boundaries(env):
    env.reset()
    env.agent_positions = torch.full((env.n_agents, 2), env.arena_radius + 1.0)
    env._enforce_boundaries()
    assert torch.all(env.agent_positions.norm(dim=1) <= env.arena_radius + 1e-5)


def test_observation_validity(env):
    obs = env.reset()["observation"]
    assert not torch.isnan(obs).any()
    assert not torch.isinf(obs).any()
    assert obs.shape == (env.n_agents, 14)


def test_rotational_dynamics(env):
    env.reset()
    env.phase = "spinning"
    env.agent_positions[:] = torch.tensor([[1.0, 0.0]] * env.n_agents)
    env._step(TensorDict({"action": torch.zeros((env.n_agents, 2))}))
    x_mean = env.agent_positions[:, 0].mean().item()
    y_mean = env.agent_positions[:, 1].mean().item()
    assert y_mean != 0.0  # rotation should have moved y component


def test_check_env_specs():
    env = MingleEnv()
    check_env_specs(env)

def test_agent_positions_shape_and_type(env):
    env.reset()
    assert isinstance(env.agent_positions, torch.Tensor)
    assert env.agent_positions.shape == (env.n_agents, 2)
    assert env.agent_positions.dtype == torch.float32


def test_room_positions_shape_and_type(env):
    env.reset()
    assert isinstance(env.room_positions, torch.Tensor)
    assert env.room_positions.shape == (env.n_rooms, 2)
    assert env.room_positions.dtype == torch.float32


def test_room_occupancy_shape_and_dtype(env):
    env.reset()
    oc = env.room_occupancy
    assert env.room_occupancy.shape == (env.n_rooms,)
    assert env.room_occupancy.dtype == torch.long


def test_action_spec_shape(env):
    spec = env.action_spec
    assert spec.shape == (env.n_agents, 2)


def test_observation_spec_shape(env):
    spec = env.observation_spec["observation"]
    assert spec.shape == (env.n_agents, 14)


def test_reset_step_consistency(env):
    obs_td = env.reset()
    assert obs_td["observation"].shape == (env.n_agents, 14)
    act = torch.zeros((env.n_agents, 2))
    out_td = env._step(TensorDict({"action": act}))
    assert out_td["observation"].shape == (env.n_agents, 14)
    assert out_td["reward"].shape == (env.n_agents, 1)


def test_done_and_terminated_flags(env):
    out_td = env.reset()
    assert out_td["done"].shape == (env.n_agents, 1)
    assert out_td["done"].dtype == torch.bool
    assert out_td["terminated"].shape == (env.n_agents, 1)
    assert out_td["terminated"].dtype == torch.bool


def test_step_does_not_crash_with_extreme_actions(env):
    env.reset()
    big_action = torch.randn((env.n_agents, 2)) * 1000
    out_td = env._step(TensorDict({"action": big_action}))
    assert out_td["reward"].shape == (env.n_agents, 1)


def test_action_clipping(env):
    env.reset()
    long_action = torch.full((env.n_agents, 2), 1000.0)
    env._step(TensorDict({"action": long_action}))
    speeds = env.agent_positions.norm(dim=1)
    assert torch.all(speeds <= env.arena_radius)


def test_compute_observations_returns_correct_shape(env):
    env.reset()
    obs = env._compute_observations()
    assert obs.shape == (env.n_agents, 14)


def test_compute_observations_no_nan_or_inf(env):
    env.reset()
    obs = env._compute_observations()
    assert not torch.isnan(obs).any()
    assert not torch.isinf(obs).any()


def test_agent_to_room_distance_within_bounds(env):
    env.reset()
    obs = env._compute_observations()
    dist_to_room = obs[:, 4]
    assert torch.all((0.0 <= dist_to_room) & (dist_to_room <= env.arena_radius + 1e-5))


def test_distance_to_center_edge_clamped(env):
    env.reset()
    obs = env._compute_observations()
    edge_dist = obs[:, 2]
    assert edge_dist.shape == (env.n_agents, )
    assert torch.all(torch.isfinite(edge_dist))


def test_nearest_agent_distance_in_range(env):
    env.reset()
    obs = env._compute_observations()
    dist_to_nearest = obs[:, 9]
    assert torch.all((0 <= dist_to_nearest) & (dist_to_nearest <= 1.0))


def test_phase_flag_valid(env):
    env.reset()
    obs = env._compute_observations()
    phase_flag = obs[:, -1]
    assert torch.all((phase_flag == 0.0) | (phase_flag == 1.0))


def test_random_seeds_produce_results():
    env = MingleEnv(n_agents=2)
    env._set_seed(42)
    positions1 = env.agent_positions.clone()
    env._set_seed(42)
    positions2 = env.agent_positions.clone()
    assert torch.allclose(positions1, positions2, atol=1e-5)


def test_set_seed_returns_seed(env):
    returned = env._set_seed(1234)
    assert isinstance(returned, int)
    assert returned == 1234


def test_step_terminated_flag_at_max_steps():
    env = MingleEnv(max_steps=3)
    env.reset()
    for _ in range(3):
        td = env._step({"action": torch.zeros((env.n_agents, 2))})
    assert td["terminated"].all()

def test_room_capacity_respected(env):
    env.reset()
    assert env.room_capacity >= 1
    assert isinstance(env.room_capacity, int)


def test_phase_mode_flag_is_valid(env):
    assert env.phase_mode in ["both", "claiming", "spinning"]


def test_spinning_duration_within_range(env):
    env.reset()
    assert env.spinning_phase_range[0] <= env.spinning_duration <= env.spinning_phase_range[1]


def test_rotate_positions_no_crash(env):
    env.reset()
    before = env.agent_positions.clone()
    env._rotate_positions(torch.pi / 4)
    after = env.agent_positions
    assert not torch.allclose(before, after)


def test_step_with_random_actions(env):
    env.reset()
    actions = torch.randn((env.n_agents, 2))
    td = env._step(TensorDict({"action": actions}))
    assert td["observation"].shape == (env.n_agents, 14)
    assert td["reward"].shape == (env.n_agents, 1)


def test_agent_stays_in_arena_after_multiple_steps(env):
    env.reset()
    for _ in range(20):
        env._step(TensorDict({"action": torch.ones((env.n_agents, 2)) * 0.5}))
    assert torch.all(env.agent_positions.norm(dim=1) <= env.arena_radius + 1e-4)


def test_negative_action_handling(env):
    env.reset()
    actions = torch.full((env.n_agents, 2), -1.0)
    env._step(TensorDict({"action": actions}))
    assert torch.all(env.agent_positions.norm(dim=1) <= env.arena_radius)


def test_dtype_consistency_in_observations(env):
    obs = env.reset()["observation"]
    assert obs.dtype == torch.float32


def test_room_distance_index_valid(env):
    env.reset()
    obs = env._compute_observations()
    room_idx = torch.argmin(torch.cdist(env.agent_positions, env.room_positions), dim=1)
    assert torch.all(room_idx >= 0)
    assert torch.all(room_idx < env.n_rooms)


def test_all_components_of_observation(env):
    env.reset()
    obs = env._compute_observations()
    # Shape already checked; now inspect ranges
    assert torch.all((obs[:, 0] >= 0) & (obs[:, 0] <= 1))  # dist to center
    assert torch.all(torch.isfinite(obs[:, 1:3]))          # dir to center
    assert torch.all(torch.isfinite(obs[:, 4:6]))          # nearest room dir


def test_reward_is_zero_without_modules(env):
    env.reset()
    reward = env._compute_rewards()
    assert torch.all(reward == 0.0)


def test_reward_module_injection():
    class DummyReward:
        def __call__(self, env):
            return torch.ones((env.n_agents, 1), device=env.device)

    env = MingleEnv(n_agents=3, reward_modules=[DummyReward()])
    env.reset()
    reward = env._compute_rewards()
    assert torch.all(reward == 1.0)


def test_step_handles_nan_actions(env):
    env.reset()
    actions = torch.full((env.n_agents, 2), float("nan"))
    td = env._step(TensorDict({"action": actions}))
    assert torch.all(torch.isfinite(td["observation"]))


def test_step_handles_inf_actions(env):
    env.reset()
    actions = torch.full((env.n_agents, 2), float("inf"))
    td = env._step(TensorDict({"action": actions}))
    assert torch.all(torch.isfinite(td["observation"]))


def test_action_spec_bounds(env):
    spec = env.action_spec
    assert torch.all(spec.low == -env.max_speed)
    assert torch.all(spec.high == env.max_speed)


def test_invalid_action_shape_raises(env):
    env.reset()
    with pytest.raises(Exception):
        env._step(TensorDict({"action": torch.randn(5)}))  # Incorrect shape


def test_step_does_not_modify_action_tensor(env):
    env.reset()
    action = torch.ones((env.n_agents, 2))
    action_copy = action.clone()
    env._step(TensorDict({"action": action}))
    assert torch.allclose(action, action_copy)


def test_reset_preserves_dtype():
    env = MingleEnv()
    td = env.reset()
    assert td["observation"].dtype == torch.float32


def test_compute_rewards_output_shape(env):
    env.reset()
    reward = env._compute_rewards()
    assert reward.shape == (env.n_agents, 1)


def test_environment_device_consistency():
    env = MingleEnv()
    env.reset()
    assert env.agent_positions.device == torch.device("cpu")


def test_rotation_preserves_norm(env):
    env.reset()
    env.agent_positions[:] = torch.tensor([[1.0, 0.0]] * env.n_agents)
    before_norms = env.agent_positions.norm(dim=1)
    env._rotate_positions(torch.pi)
    after_norms = env.agent_positions.norm(dim=1)
    assert torch.allclose(before_norms, after_norms, atol=1e-5)


def test_step_with_alternating_phases():
    env = MingleEnv()
    env.reset()
    env.phase_mode = "both"
    env.spinning_duration = 1
    env._step(TensorDict({"action": torch.zeros((env.n_agents, 2))}))
    assert env.phase == "claiming"


def test_observation_contains_velocity(env):
    env.reset()
    obs = env._compute_observations()
    vel_x = obs[:, 8]
    vel_y = obs[:, 9]
    assert torch.all(torch.isfinite(vel_x))
    assert torch.all(torch.isfinite(vel_y))


def test_all_phases_run_without_crash():
    for phase in ["spinning", "claiming"]:
        env = MingleEnv()
        env.phase = phase
        env.reset()
        env._step(TensorDict({"action": torch.zeros((env.n_agents, 2))}))


def test_all_phase_modes_valid():
    for mode in ["both", "claiming", "spinning"]:
        env = MingleEnv(phase_mode=mode)
        env.reset()
        assert env.phase_mode == mode


def test_environment_repr_does_not_crash(env):
    repr_str = repr(env)
    assert isinstance(repr_str, str)
    assert "MingleEnv" in repr_str

def test_collision_rate_metric(env):
    metric = CollisionRateMetric(min_distance=0.5)
    metric.reset()
    for _ in range(5):
        metric.update(env)
    result = metric.compute()
    assert "collision_rate" in result
    assert 0 <= result["collision_rate"] <= 1

def test_room_occupancy_rate_metric(env):
    metric = RoomOccupancyRateMetric()
    metric.reset()
    # If no room positions, update should not fail
    env.room_positions = None
    metric.update(env)  # no error expected

    # Provide room positions for meaningful test
    env.room_positions = torch.randn(env.n_rooms, 2)
    for _ in range(5):
        metric.update(env)
    result = metric.compute()
    assert "room_occupancy_rate" in result
    assert 0 <= result["room_occupancy_rate"] <= 1

def test_center_presence_metric(env):
    metric = CenterPresenceMetric()
    metric.reset()
    env.center_radius = 1.0  # ensure attribute exists
    for _ in range(5):
        metric.update(env)
    result = metric.compute()
    assert "center_presence_rate" in result
    assert 0 <= result["center_presence_rate"] <= 1

def test_average_step_distance_metric(env):
    metric = AverageStepDistanceMetric()
    metric.reset()
    env.agent_positions = torch.randn(env.n_agents, 2)
    metric.update(env)  # first update sets last_positions but no distance
    for _ in range(5):
        # simulate agent movement
        env.agent_positions += torch.randn(env.n_agents, 2) * 0.1
        metric.update(env)
    result = metric.compute()
    assert "average_step_distance" in result
    assert result["average_step_distance"] >= 0

def test_idle_agent_rate_metric(env):
    metric = IdleAgentRateMetric(threshold=0.01)
    metric.reset()
    env.agent_positions = torch.randn(env.n_agents, 2)
    metric.update(env)  # initialize last_positions
    for _ in range(5):
        # simulate small movements below threshold
        env.agent_positions += torch.randn(env.n_agents, 2) * 0.001
        metric.update(env)
    result = metric.compute()
    assert "idle_agent_rate" in result
    assert 0 <= result["idle_agent_rate"] <= 1

def test_room_switches_metric(env):
    metric = RoomSwitchesMetric()
    metric.reset()
    env.room_positions = torch.randn(env.n_rooms, 2)
    metric.update(env)  # init last_closest
    for _ in range(5):
        # simulate random agent movements to trigger switches
        env.agent_positions += torch.randn(env.n_agents, 2) * 0.5
        metric.update(env)
    result = metric.compute()
    assert "room_switches" in result
    assert isinstance(result["room_switches"], int)
    assert result["room_switches"] >= 0

def test_phase_time_metric(env):
    metric = PhaseTimeMetric()
    metric.reset()
    # simulate phases for env.phase attribute
    for phase in ["spinning", "claiming", "spinning", "claiming", "spinning"]:
        env.phase = phase
        metric.update(env)
    result = metric.compute()
    assert "phase_time_spinning" in result
    assert "phase_time_claiming" in result
    total = sum(result.values())
    # distribution should sum approximately to 1.0
    assert 0.99 <= total <= 1.01

def test_agent_density_metric(env):
    metric = AgentDensityMetric(radius=1.0)
    metric.reset()
    for _ in range(5):
        env.agent_positions = torch.randn(env.n_agents, 2) * 0.5
        metric.update(env)
    result = metric.compute()
    assert "agent_density" in result
    assert result["agent_density"] >= 0

def test_max_distance_from_center_metric(env):
    metric = MaxDistanceFromCenterMetric()
    metric.reset()
    env.agent_positions = torch.randn(env.n_agents, 2) * 3
    for _ in range(5):
        metric.update(env)
    result = metric.compute()
    assert "max_distance_from_center" in result
    assert result["max_distance_from_center"] > 0

def test_min_agent_distance_metric(env):
    metric = MinAgentDistanceMetric()
    metric.reset()
    env.agent_positions = torch.randn(env.n_agents, 2)
    for _ in range(5):
        metric.update(env)
    result = metric.compute()
    assert "min_agent_distance" in result
    assert result["min_agent_distance"] >= 0

def test_average_room_distance_metric(env):
    metric = AverageRoomDistanceMetric()
    metric.reset()
    env.room_positions = torch.randn(env.n_rooms, 2)
    for _ in range(5):
        env.agent_positions = torch.randn(env.n_agents, 2)
        metric.update(env)
    result = metric.compute()
    assert "average_room_distance" in result
    assert result["average_room_distance"] >= 0

def test_agent_movement_variance_metric(env):
    metric = AgentMovementVarianceMetric()
    metric.reset()
    env.agent_positions = torch.randn(env.n_agents, 2)
    metric.update(env)
    for _ in range(5):
        env.agent_positions += torch.randn(env.n_agents, 2) * 0.1
        metric.update(env)
    result = metric.compute()
    assert "agent_movement_variance" in result
    assert result["agent_movement_variance"] >= 0