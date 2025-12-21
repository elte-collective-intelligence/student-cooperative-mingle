# src/envs/transforms/fairness_reward_transform.py

from typing import Dict
import torch
from tensordict import TensorDictBase


def make_fairness_reward_transform(fairness_cfg: Dict):
    """
    Returns a callable that can be added to TransformedEnv.
    This callable receives the tensordict at each step and
    redistributes the 'reward' based on the current fairness mode.

    Supported modes:
        - "none": No redistribution
        - "gini": GINI-coefficient-based reward redistribution (inverse reward weighting)
        - "participation_variance": Variance in participation penalizes unequal engagement
        - "exclusion_counts": Tracks and compensates excluded agents
    """

    mode = fairness_cfg.get("mode", "none")
    alpha = float(fairness_cfg.get("alpha", 0.5))

    participation_threshold = float(fairness_cfg.get("participation_threshold", 0.0))
    exclusion_threshold = float(fairness_cfg.get("exclusion_threshold", 0.0))

    # Episode-wide state (stored in closure)
    cum_reward = None            # (n_agents, 1)
    participation_counts = None  # (n_agents, 1)
    exclusion_counts = None      # (n_agents, 1)
    steps = 0

    def gini(x: torch.Tensor) -> torch.Tensor:
        """Scalar Gini coefficient for a vector (x)."""
        x_flat = x.view(-1)
        if x_flat.numel() == 0:
            return x_flat.new_tensor(0.0)

        # Make non-negative (more convenient for Gini)
        min_x = x_flat.min()
        if min_x < 0:
            x_flat = x_flat - min_x

        mean_x = x_flat.mean()
        if mean_x == 0:
            return x_flat.new_tensor(0.0)

        sorted_x, _ = torch.sort(x_flat)
        n = x_flat.numel()
        idx = torch.arange(1, n + 1, dtype=sorted_x.dtype, device=sorted_x.device)

        # G = (2 * sum_i i*x_i) / (n * sum x) - (n+1)/n
        G = (2.0 * (idx * sorted_x).sum()) / (n * sorted_x.sum()) - (n + 1.0) / n
        return G.clamp(0.0, 1.0)

    def fairness_transform(td: TensorDictBase) -> TensorDictBase:
        nonlocal cum_reward, participation_counts, exclusion_counts, steps

        # RESET: no "reward" key yet in td
        if "reward" not in td.keys():
            # Try to infer n_agents from observation
            obs = td.get("observation", None)
            if obs is not None:
                # In Mingle, shape is (n_agents, obs_dim), so -2 dim is n_agents
                if obs.ndim >= 2:
                    n_agents = obs.shape[-2]
                else:
                    n_agents = 1
                device = obs.device
                shape = (n_agents, 1)
            elif cum_reward is not None:
                shape = cum_reward.shape
                device = cum_reward.device
            else:
                # Edge case, but don't throw error
                return td

            cum_reward = torch.zeros(shape, device=device)
            participation_counts = torch.zeros(shape, device=device)
            exclusion_counts = torch.zeros(shape, device=device)
            steps = 0
            return td

        # STEP: reward exists
        reward = td.get("reward")
        if reward is None:
            return td

        # Assume unbatched env -> (n_agents, 1)
        if cum_reward is None or cum_reward.shape != reward.shape or cum_reward.device != reward.device:
            cum_reward = torch.zeros_like(reward)
            participation_counts = torch.zeros_like(reward)
            exclusion_counts = torch.zeros_like(reward)
            steps = 0

        steps += 1
        cum_reward = cum_reward + reward

        # 1) No redistribution
        if mode == "none":
            return td.set("reward", reward)

        # 2) Gini-based redistribution
        if mode == "gini":
            base = cum_reward.detach()

            # Those with less cumulative reward get more
            abs_cum = base.abs() + 1e-6
            inv = 1.0 / abs_cum
            inv_sum = inv.sum()
            if inv_sum <= 0:
                return td.set("reward", reward)

            weights = inv / inv_sum                        # (n_agents, 1)
            total = reward.sum()                           # scalar
            redistributed = total * weights                # (n_agents, 1)

            G = gini(base)
            alpha_eff = float(alpha) * float(G.item())     # stronger when inequality is high
            alpha_eff = max(0.0, min(alpha_eff, 1.0))

            new_reward = (1.0 - alpha_eff) * reward + alpha_eff * redistributed
            return td.set("reward", new_reward)

        # 3) Participation variance based redistribution
        if mode == "participation_variance":
            # Simple proxy: participation = reward > threshold
            participated = (reward > participation_threshold).float()
            participation_counts = participation_counts + participated

            # participation rate ~ [0,1], over the episode
            p = participation_counts / float(max(1, steps))
            mean_p = p.mean()
            var_p = ((p - mean_p) ** 2).mean()

            # Those below average get more
            diff = (mean_p - p)
            weights_raw = torch.clamp(diff, min=0.0)
            raw_sum = weights_raw.sum()
            if raw_sum <= 0:
                return td.set("reward", reward)

            weights = weights_raw / raw_sum
            total = reward.sum()
            redistributed = total * weights

            # participation rate in [0,1] -> max var ~ 0.25 (half 0, half 1)
            max_var = 0.25
            norm_var = torch.clamp(var_p / max_var, 0.0, 1.0)
            alpha_eff = float(alpha) * float(norm_var.item())

            new_reward = (1.0 - alpha_eff) * reward + alpha_eff * redistributed
            return td.set("reward", new_reward)

        # 4) Exclusion counts based redistribution
        if mode == "exclusion_counts":
            # Simple proxy: exclusion = reward < threshold (e.g., large negative penalty)
            excluded = (reward < exclusion_threshold).float()
            exclusion_counts = exclusion_counts + excluded

            weights_raw = exclusion_counts.clone()
            raw_sum = weights_raw.sum()
            if raw_sum <= 0:
                return td.set("reward", reward)

            weights = weights_raw / raw_sum
            total = reward.sum()
            redistributed = total * weights

            # Intensity: how unequal are the exclusions
            mean_c = exclusion_counts.mean()
            var_c = ((exclusion_counts - mean_c) ** 2).mean()
            if mean_c > 0:
                imbalance = torch.clamp(var_c / (mean_c ** 2 + 1e-6), 0.0, 1.0)
            else:
                imbalance = reward.new_tensor(0.0)

            alpha_eff = float(alpha) * float(imbalance.item())
            alpha_eff = max(0.0, min(alpha_eff, 1.0))

            new_reward = (1.0 - alpha_eff) * reward + alpha_eff * redistributed
            return td.set("reward", new_reward)

        # Unknown mode: don't break the env, just leave unchanged
        return td.set("reward", reward)

    return fairness_transform
