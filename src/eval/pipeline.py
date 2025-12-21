import os
import json
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.distributions.continuous import TanhNormal
from tensordict import TensorDict

from src.envs.modules.metric_module import (
    CollisionRateMetric, RoomOccupancyRateMetric, CenterPresenceMetric,
    AverageStepDistanceMetric, IdleAgentRateMetric, RoomSwitchesMetric,
    PhaseTimeMetric, AgentDensityMetric,
    MaxDistanceFromCenterMetric, MinAgentDistanceMetric, AverageRoomDistanceMetric, AgentMovementVarianceMetric,
    EpisodeSuccessMetric, GiniFairnessMetric, ClaimingPhaseEfficiencyMetric,
    # Communication and fairness metrics
    MessageUsageMetric, CommunicationEffectivenessMetric, ParticipationFairnessMetric
)

def evaluate(policy_module, env, device, num_episodes=10, max_steps=300, out_dir=None):
    print(f"\n[EVAL] Running evaluation...\n")

    policy_module.eval()
    policy_module.to(device)
    env.to(device)

    metric_modules = [
        # Core environment metrics
        CollisionRateMetric(), RoomOccupancyRateMetric(), CenterPresenceMetric(),
        AverageStepDistanceMetric(), IdleAgentRateMetric(), RoomSwitchesMetric(),
        PhaseTimeMetric(), AgentDensityMetric(),
        MaxDistanceFromCenterMetric(),
        MinAgentDistanceMetric(),
        AverageRoomDistanceMetric(),
        AgentMovementVarianceMetric(),
        # Success and fairness metrics (Task 2)
        EpisodeSuccessMetric(),
        GiniFairnessMetric(),
        ClaimingPhaseEfficiencyMetric(),
        ParticipationFairnessMetric(),
        # Communication metrics (Task 1 & 4)
        MessageUsageMetric(),
        CommunicationEffectivenessMetric(),
    ]

    metrics_over_episodes = defaultdict(list)

    for episode in range(num_episodes):
        td = env._reset()
        for metric in metric_modules:
            metric.reset()

        for step in range(max_steps):
            observation = td.select("observation")

            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                td_action = policy_module(observation)
                action_dist = TanhNormal(td_action["loc"], td_action["scale"])
                actions = action_dist.sample()

            td = env._step(TensorDict({"action": actions}, batch_size=[]))

            for metric in metric_modules:
                metric.update(env)

        episode_results = {k: v for metric in metric_modules for k, v in metric.compute().items()}
        for k, v in episode_results.items():
            metrics_over_episodes[k].append(v)

        print(f"[EPISODE] {episode + 1}/{num_episodes} results:")
        for k, v in episode_results.items():
            print(f"    {k}: {v:.3f}")
        print()

    # Create output directory
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("eval_results", f"eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics to JSON
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_over_episodes, f, indent=2)
    print(f"[SAVE] Metrics saved to {out_dir}/metrics.json\n")

    # Plot metrics
    for metric, values in metrics_over_episodes.items():
        plt.figure()
        plt.plot(values, marker="o")
        plt.title(f"{metric.replace('_', ' ').title()} Across Episodes")
        plt.xlabel("Episode")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}.png"))
        plt.close()
    print(f"[PLOT] Plots saved to: {out_dir}\n")

    # Print summary
    print("[SUMMARY] Evaluation Summary:")
    for k, v in metrics_over_episodes.items():
        mean_val = sum(v) / len(v)
        std_val = (sum((x - mean_val) ** 2 for x in v) / len(v)) ** 0.5
        print(f"""
[METRIC] {k}:
    Mean = {mean_val:.4f}
    Std  = {std_val:.4f}
""")

    return metrics_over_episodes
