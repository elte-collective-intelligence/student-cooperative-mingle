import torch

from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import EfficiencyReward, FairnessReward, MultiObjectiveReward
from src.envs.modules.metric_module import JainFairnessMetric, ParticipationRangeMetric


class DummyEnv:
    def __init__(self):
        self.n_agents = 4
        self.n_rooms = 2
        self.room_capacity = 2
        self.device = torch.device("cpu")
        self.phase = "claiming"
        self.current_step = 1

        self.agent_positions = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
        ], dtype=torch.float32)

        self.room_positions = torch.tensor([
            [0.0, 0.0],
            [5.0, 5.0],
        ], dtype=torch.float32)

        self.room_radius = 1.0
        self.arena_radius = 10.0
        self.agent_in_room = torch.tensor([True, True, True, False])
        self.forced_to_leave = torch.tensor([False, False, False, False])


def _place_agents_in_rooms(env: MingleEnv) -> None:
    env.reset()
    positions = []
    for i in range(env.n_agents):
        room_idx = i % env.n_rooms
        room_pos = env.room_positions[room_idx]
        positions.append(room_pos + torch.tensor([0.1, 0.0], device=env.device))
    env.agent_positions = torch.stack(positions, dim=0)
    env.phase = "claiming"


def test_participation_range_metric_tracks_unequal_participation():
    env = DummyEnv()
    metric = ParticipationRangeMetric()

    metric.update(env)
    result = metric.compute()

    assert "participation_range" in result
    assert result["participation_range"] > 0.0
    assert result["max_participation_rate"] >= result["min_participation_rate"]


def test_multi_objective_scalarization_matches_components():
    env = MingleEnv(n_agents=4, n_rooms=2, phase_mode="claiming")
    _place_agents_in_rooms(env)

    alpha = 0.75

    eff_expected = EfficiencyReward(phase_mode="claiming")
    fair_expected = FairnessReward(phase_mode="claiming", fairness_metric="jain")
    eff_expected._activate()
    fair_expected._activate()
    expected = alpha * eff_expected(env) + (1.0 - alpha) * fair_expected(env)

    efficiency = EfficiencyReward(phase_mode="claiming")
    fairness = FairnessReward(phase_mode="claiming", fairness_metric="jain")
    efficiency._activate()
    fairness._activate()

    scalarized = MultiObjectiveReward(
        alpha=alpha,
        efficiency_module=efficiency,
        fairness_module=fairness,
        phase_mode="claiming",
    )
    scalarized._activate()

    actual = scalarized(env)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_jain_fairness_metric_equal_rewards():
    env = MingleEnv(n_agents=4, n_rooms=2, phase_mode="claiming")
    env.reset()
    env.last_rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]])

    metric = JainFairnessMetric()
    metric.update(env)
    result = metric.compute()

    assert abs(result["jain_index"] - 1.0) < 1e-6


def test_multi_objective_reward_shape_is_correct():
    env = DummyEnv()
    efficiency = EfficiencyReward(phase_mode="claiming")
    fairness = FairnessReward(phase_mode="claiming", fairness_metric="participation_range")
    reward_module = MultiObjectiveReward(
        alpha=0.5,
        efficiency_module=efficiency,
        fairness_module=fairness,
        phase_mode="claiming",
    )
    reward_module._activate()

    reward = reward_module(env)

    assert reward.shape == (env.n_agents, 1)
    assert torch.isfinite(reward).all()


def test_pareto_alpha_changes_reward_tradeoff_shape():
    env = DummyEnv()

    efficiency_only = MultiObjectiveReward(
        alpha=1.0,
        efficiency_module=EfficiencyReward(phase_mode="claiming"),
        fairness_module=FairnessReward(phase_mode="claiming", fairness_metric="participation_range"),
        phase_mode="claiming",
    )
    fairness_only = MultiObjectiveReward(
        alpha=0.0,
        efficiency_module=EfficiencyReward(phase_mode="claiming"),
        fairness_module=FairnessReward(phase_mode="claiming", fairness_metric="participation_range"),
        phase_mode="claiming",
    )
    efficiency_only._activate()
    fairness_only._activate()

    efficiency_reward = efficiency_only(env)
    fairness_reward = fairness_only(env)

    assert efficiency_reward.shape == fairness_reward.shape
    assert torch.isfinite(efficiency_reward).all()
    assert torch.isfinite(fairness_reward).all()
