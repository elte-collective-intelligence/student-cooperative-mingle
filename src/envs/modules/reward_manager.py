from typing import List
import torch

from src.envs.modules.reward_module import GetToRoomReward, StayInRoomReward, CollisionAvoidanceReward, RewardModule, InsideCenterReward

from typing import List
import torch

from src.envs.modules.reward_module import GetToRoomReward, StayInRoomReward, CollisionAvoidanceReward, RewardModule, InsideCenterReward

def select_reward_modules():
    available_modules = {
        "InsideCenterReward":           InsideCenterReward,
        "CollisionAvoidanceReward":     CollisionAvoidanceReward,
        "GetToRoomReward":              GetToRoomReward,
        "StayInRoomReward":             StayInRoomReward,
    }

    selected = {"spinning": [], "claiming": []}
    thresholds = {"spinning": [], "claiming": []}

    for phase in selected.keys():
        print(f"\n➡️  Selecting reward modules for phase: {phase.upper()}\n")

        available_keys = list(available_modules.keys())

        while True:
            print("Available modules:")
            for i, name in enumerate(available_keys):
                print(f"  [{i}] {name}")

            print("\nSelected so far:")
            for i, mod in enumerate(selected[phase]):
                mod_name = mod.__class__.__name__
                if i == 0:
                    print(f"  {mod_name} (active from start)")
                else:
                    print(f"  {mod_name} @ mean_reward > {thresholds[phase][i-1]:.4f}")

            choice = input("\nSelect a module index (or 'done'): ").strip()

            if choice.lower() == "done":
                print(f"\n✅ Finished selection for phase: {phase.upper()}\n")
                break

            try:
                idx = int(choice)
                if idx < 0 or idx >= len(available_keys):
                    raise IndexError

                name = available_keys[idx]
                module = available_modules[name](phase_mode=phase)
                selected[phase].append(module)

                threshold = float(input(f"Enter mean reward threshold to activate next_module: ").strip())
                thresholds[phase].append(threshold)
                print(f"\n➕ Added {name} | mean_reward for next module > {threshold:.4f}\n")

            except (ValueError, IndexError):
                print("\n❌ Invalid input. Please try again.\n")

    return selected, thresholds


class RewardManager:
    def __init__(self, reward_modules: List[RewardModule], thresholds: List[float], phase: str):
        self.reward_modules = reward_modules
        self.thresholds = thresholds
        self.phase = phase
        self.current_idx = 0
        self.reward_modules[0].active = True

    def update(self, reward_mean: float):
        """
        Activates next reward module if current mean reward surpasses the corresponding threshold.
        """
        if self.current_idx < len(self.thresholds):
            threshold = self.thresholds[self.current_idx]
            print(reward_mean, threshold)
            if reward_mean > threshold:
                self.current_idx += 1
                self.reward_modules[self.current_idx].active = True
                print(f"🆕 Activated new reward: {type(self.reward_modules[self.current_idx]).__name__} "
                      f"for phase: {self.phase} (mean_reward={reward_mean:.4f} > {threshold:.4f})")

    def __call__(self, env):
        total_reward = torch.zeros((env.n_agents, 1), device=env.device)
        for module in self.reward_modules:
            if module.active:
                total_reward += module(env)
        return total_reward