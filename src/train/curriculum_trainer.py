"""
Curriculum Learning Trainer for Cooperative Mingle

Implements progressive difficulty scaling:
- Start: fewer agents, more rooms, larger capacities (easy)
- End: more agents, fewer rooms, smaller capacities (hard)

This helps agents learn basic coordination before tackling harder scenarios.

Usage:
    python -m src.train.curriculum_trainer
"""

import os
import json
import time
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tensordict import TensorDict

from src.envs.mingle_env import MingleEnv
from src.utils.config import load_and_merge_configs


@dataclass
class CurriculumStage:
    """Defines a curriculum stage with specific difficulty parameters."""
    name: str
    n_agents: int
    n_rooms: int
    room_capacity: int
    frames: int  # Training frames for this stage
    success_threshold: float = 0.6  # Required success rate to advance


# Default curriculum: Easy -> Medium -> Hard
DEFAULT_CURRICULUM = [
    CurriculumStage(
        name="easy",
        n_agents=2,
        n_rooms=4,
        room_capacity=3,
        frames=1000,
        success_threshold=0.5,
    ),
    CurriculumStage(
        name="medium",
        n_agents=4,
        n_rooms=3,
        room_capacity=2,
        frames=1500,
        success_threshold=0.5,
    ),
    CurriculumStage(
        name="hard",
        n_agents=6,
        n_rooms=2,
        room_capacity=2,
        frames=2000,
        success_threshold=0.4,
    ),
]


class CurriculumTrainer:
    """
    Curriculum Learning Trainer.

    Progressively increases task difficulty as the agent learns.
    Tracks adaptation speed and learning stability across stages.
    """

    def __init__(
        self,
        curriculum: List[CurriculumStage] = None,
        device: str = "cpu",
        base_config: Dict = None,
    ):
        self.curriculum = curriculum or DEFAULT_CURRICULUM
        self.device = torch.device(device)
        self.base_config = base_config or {}

        # Metrics tracking
        self.stage_metrics: Dict[str, Dict] = {}
        self.adaptation_times: List[float] = []
        self.stage_transitions: List[Dict] = []

    def create_env_for_stage(self, stage: CurriculumStage) -> MingleEnv:
        """Create environment configured for a specific curriculum stage."""
        return MingleEnv(
            n_agents=stage.n_agents,
            n_rooms=stage.n_rooms,
            room_capacity=stage.room_capacity,
            max_steps=100,
            arena_radius=10.0,
            center_radius=3.0,
            phase_mode="both",
        )

    def evaluate_stage_performance(
        self,
        env: MingleEnv,
        num_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate current performance on a stage.

        Returns dict with:
        - success_rate: fraction of episodes where all agents find rooms
        - mean_reward: average episode reward
        - collision_rate: average collision rate
        """
        metrics = {
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "collision_rate": 0.0,
            "room_occupancy": 0.0,
        }

        episode_rewards = []
        successes = 0
        total_collisions = 0
        total_checks = 0
        total_in_room = 0
        total_agents = 0

        for _ in range(num_episodes):
            td = env.reset()
            episode_reward = 0

            for _ in range(env.max_steps):
                # Random policy for evaluation
                actions = torch.randn(env.n_agents, 2) * 0.3
                actions = actions.clamp(-env.max_speed, env.max_speed)

                step_td = TensorDict({"action": actions}, batch_size=[])
                td = env._step(step_td)

                episode_reward += td["reward"].sum().item()

                # Check collisions
                dists = torch.cdist(env.agent_positions, env.agent_positions)
                mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
                collisions = ((dists < 0.5) & mask).sum().item()
                total_collisions += collisions
                total_checks += mask.sum().item()

                # Check room occupancy
                if env.room_positions is not None:
                    room_dists = torch.cdist(env.agent_positions, env.room_positions)
                    in_room = (room_dists < env.room_radius).any(dim=1)
                    total_in_room += in_room.sum().item()
                    total_agents += env.n_agents

                if td["done"].any() or td["terminated"].any():
                    break

            episode_rewards.append(episode_reward)

            # Check success: all agents in valid rooms
            if env.room_positions is not None:
                room_dists = torch.cdist(env.agent_positions, env.room_positions)
                in_room = (room_dists < env.room_radius).any(dim=1)
                if in_room.all():
                    successes += 1

        metrics["success_rate"] = successes / num_episodes
        metrics["mean_reward"] = np.mean(episode_rewards)
        metrics["collision_rate"] = total_collisions / max(total_checks, 1)
        metrics["room_occupancy"] = total_in_room / max(total_agents, 1)

        return metrics

    def train_stage(
        self,
        stage: CurriculumStage,
        stage_idx: int,
    ) -> Dict[str, Any]:
        """
        Train on a single curriculum stage.

        Returns metrics for the stage including:
        - rewards over time
        - final performance
        - time to reach threshold
        """
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage_idx + 1}/{len(self.curriculum)}: {stage.name.upper()}")
        print(f"{'='*60}")
        print(f"  Agents: {stage.n_agents}")
        print(f"  Rooms: {stage.n_rooms}")
        print(f"  Capacity: {stage.room_capacity}")
        print(f"  Target frames: {stage.frames}")
        print(f"  Success threshold: {stage.success_threshold}")

        env = self.create_env_for_stage(stage)

        # Training metrics
        rewards = []
        success_rates = []
        frames_to_threshold = None
        start_time = time.time()

        # Simple training loop (random policy baseline for curriculum demonstration)
        # In production, this would use PPO/IPPO/MAPPO
        td = env.reset()
        episode_reward = 0
        frame = 0
        eval_interval = stage.frames // 10

        while frame < stage.frames:
            # Random actions (replace with actual policy in production)
            actions = torch.randn(env.n_agents, 2) * 0.3
            actions = actions.clamp(-env.max_speed, env.max_speed)

            step_td = TensorDict({"action": actions}, batch_size=[])
            td = env._step(step_td)

            episode_reward += td["reward"].sum().item()
            frame += env.n_agents

            if td["done"].any() or td["terminated"].any():
                rewards.append(episode_reward)
                episode_reward = 0
                td = env.reset()

            # Periodic evaluation
            if frame % eval_interval == 0 and frame > 0:
                eval_metrics = self.evaluate_stage_performance(env, num_episodes=5)
                success_rates.append(eval_metrics["success_rate"])

                print(f"  Frame {frame}/{stage.frames} | "
                      f"Success: {eval_metrics['success_rate']:.2%} | "
                      f"Reward: {eval_metrics['mean_reward']:.2f}")

                # Check if threshold reached
                if frames_to_threshold is None and eval_metrics["success_rate"] >= stage.success_threshold:
                    frames_to_threshold = frame
                    print(f"  >> Threshold reached at frame {frame}!")

        # Final evaluation
        final_metrics = self.evaluate_stage_performance(env, num_episodes=10)

        stage_time = time.time() - start_time

        stage_result = {
            "stage_name": stage.name,
            "stage_idx": stage_idx,
            "config": {
                "n_agents": stage.n_agents,
                "n_rooms": stage.n_rooms,
                "room_capacity": stage.room_capacity,
            },
            "rewards": rewards,
            "success_rates": success_rates,
            "final_metrics": final_metrics,
            "frames_to_threshold": frames_to_threshold,
            "total_frames": frame,
            "training_time": stage_time,
            "threshold_reached": final_metrics["success_rate"] >= stage.success_threshold,
        }

        print(f"\n  Stage Complete:")
        print(f"    Final success rate: {final_metrics['success_rate']:.2%}")
        print(f"    Final mean reward: {final_metrics['mean_reward']:.2f}")
        print(f"    Frames to threshold: {frames_to_threshold or 'Not reached'}")
        print(f"    Training time: {stage_time:.1f}s")

        return stage_result

    def run_curriculum(self) -> Dict[str, Any]:
        """
        Run the full curriculum training.

        Returns comprehensive results including:
        - Per-stage metrics
        - Adaptation speed
        - Learning stability
        """
        print("\n" + "="*60)
        print("CURRICULUM LEARNING TRAINER")
        print("="*60)
        print(f"Total stages: {len(self.curriculum)}")
        print(f"Device: {self.device}")

        all_results = {
            "stages": [],
            "curriculum_config": [
                {
                    "name": s.name,
                    "n_agents": s.n_agents,
                    "n_rooms": s.n_rooms,
                    "room_capacity": s.room_capacity,
                    "frames": s.frames,
                }
                for s in self.curriculum
            ],
        }

        total_start = time.time()

        for idx, stage in enumerate(self.curriculum):
            stage_result = self.train_stage(stage, idx)
            all_results["stages"].append(stage_result)

            # Track adaptation
            if stage_result["frames_to_threshold"]:
                self.adaptation_times.append(stage_result["frames_to_threshold"])

            self.stage_transitions.append({
                "from_stage": idx,
                "to_stage": idx + 1 if idx < len(self.curriculum) - 1 else None,
                "threshold_reached": stage_result["threshold_reached"],
            })

        total_time = time.time() - total_start

        # Compute summary metrics
        all_results["summary"] = {
            "total_training_time": total_time,
            "stages_completed": len(self.curriculum),
            "stages_passed": sum(1 for s in all_results["stages"] if s["threshold_reached"]),
            "mean_adaptation_frames": np.mean(self.adaptation_times) if self.adaptation_times else None,
            "adaptation_times": self.adaptation_times,
        }

        print("\n" + "="*60)
        print("CURRICULUM COMPLETE")
        print("="*60)
        print(f"Total time: {total_time:.1f}s")
        print(f"Stages passed: {all_results['summary']['stages_passed']}/{len(self.curriculum)}")
        if self.adaptation_times:
            print(f"Mean adaptation frames: {all_results['summary']['mean_adaptation_frames']:.0f}")

        return all_results

    def save_results(self, results: Dict, output_dir: str = "curriculum_results"):
        """Save curriculum results and generate plots."""
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"curriculum_results_{timestamp}.json")

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(json_path, "w") as f:
            json.dump(convert(results), f, indent=2)
        print(f"Results saved to {json_path}")

        # Generate plots
        self.plot_curriculum_progress(results, output_dir, timestamp)

    def plot_curriculum_progress(self, results: Dict, output_dir: str, timestamp: str):
        """Generate curriculum learning progress plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        stages = results["stages"]
        stage_names = [s["stage_name"] for s in stages]

        # Plot 1: Success Rate per Stage
        ax = axes[0, 0]
        final_success = [s["final_metrics"]["success_rate"] for s in stages]
        thresholds = [self.curriculum[i].success_threshold for i in range(len(stages))]

        x = np.arange(len(stages))
        width = 0.35
        ax.bar(x - width/2, final_success, width, label="Achieved", color="steelblue")
        ax.bar(x + width/2, thresholds, width, label="Threshold", color="coral", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in stage_names])
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate by Stage")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Adaptation Speed (Frames to Threshold)
        ax = axes[0, 1]
        frames_to_thresh = [s["frames_to_threshold"] or s["total_frames"] for s in stages]
        colors = ["green" if s["threshold_reached"] else "red" for s in stages]
        ax.bar(x, frames_to_thresh, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in stage_names])
        ax.set_ylabel("Frames to Threshold")
        ax.set_title("Adaptation Speed")
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: Difficulty Progression
        ax = axes[1, 0]
        # Compute difficulty score: more agents + fewer rooms + smaller capacity = harder
        difficulties = []
        for s in self.curriculum:
            difficulty = s.n_agents / s.n_rooms / s.room_capacity
            difficulties.append(difficulty)

        ax.plot(x, difficulties, 'o-', color="purple", linewidth=2, markersize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in stage_names])
        ax.set_ylabel("Difficulty Score")
        ax.set_title("Curriculum Difficulty Progression")
        ax.grid(alpha=0.3)

        # Plot 4: Learning Curves (Success Rate over Frames)
        ax = axes[1, 1]
        for i, stage in enumerate(stages):
            if stage["success_rates"]:
                # Create x-axis for this stage
                stage_frames = np.linspace(0, stage["total_frames"], len(stage["success_rates"]))
                offset = sum(s["total_frames"] for s in stages[:i])
                ax.plot(stage_frames + offset, stage["success_rates"],
                       label=stage["stage_name"].upper(), linewidth=2)

        ax.set_xlabel("Total Frames")
        ax.set_ylabel("Success Rate")
        ax.set_title("Learning Curves Across Curriculum")
        ax.legend()
        ax.grid(alpha=0.3)

        # Add vertical lines for stage boundaries
        cumulative_frames = 0
        for i, stage in enumerate(stages[:-1]):
            cumulative_frames += stage["total_frames"]
            ax.axvline(x=cumulative_frames, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"curriculum_progress_{timestamp}.png"), dpi=150)
        plt.close()
        print(f"Plot saved to {output_dir}/curriculum_progress_{timestamp}.png")


def main():
    """Run curriculum learning trainer."""
    print("\n" + "="*60)
    print("COOPERATIVE MINGLE - CURRICULUM LEARNING")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create trainer with default curriculum
    trainer = CurriculumTrainer(
        curriculum=DEFAULT_CURRICULUM,
        device=device,
    )

    # Run curriculum
    results = trainer.run_curriculum()

    # Save results
    trainer.save_results(results)

    print("\nCurriculum learning complete!")


if __name__ == "__main__":
    main()
