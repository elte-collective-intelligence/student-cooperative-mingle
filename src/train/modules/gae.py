from torchrl.objectives.value import GAE
from tensordict import TensorDictBase
import torch


class RunningMeanStd:
    """
    Running mean and standard deviation tracker for reward normalization.
    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon  # Small epsilon to prevent division by zero

    def update(self, x):
        """Update running statistics with new batch of data."""
        batch_mean = x.mean()
        batch_var = x.var() if x.numel() > 1 else torch.tensor(0.0)
        batch_count = x.numel()

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics (Welford's algorithm)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        """Normalize input using running statistics."""
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)


class StableGAE(GAE):
    """
    A more stable variant of the Generalized Advantage Estimation (GAE) module.

    Improvements over base GAE:
        1. Running reward normalization to handle varying reward scales
        2. Advantage normalization to zero mean and unit variance
        3. Value target clamping to prevent extreme TD targets
        4. Gradient-friendly operations throughout
        5. Optional reward clipping for robustness

    Parameters:
        gamma (float): Discount factor.
        lmbda (float): Lambda parameter for GAE.
        value_network (ValueOperator): The critic network to estimate state values.
        average_gae (bool): If True, averages advantages over all samples.
        normalize_rewards (bool): If True, normalizes rewards using running statistics.
        clip_rewards (float): If > 0, clips rewards to [-clip_rewards, clip_rewards].
        value_clip_range (float): Clamp value targets to [-range, range].
    """

    def __init__(
        self,
        gamma: float,
        lmbda: float,
        value_network,
        average_gae: bool = True,
        normalize_rewards: bool = True,
        clip_rewards: float = 10.0,
        value_clip_range: float = 100.0,
    ):
        super().__init__(
            gamma=gamma,
            lmbda=lmbda,
            value_network=value_network,
            average_gae=average_gae,
        )
        self.normalize_rewards = normalize_rewards
        self.clip_rewards = clip_rewards
        self.value_clip_range = value_clip_range

        # Running statistics for reward normalization
        self.reward_stats = RunningMeanStd()
        self.return_stats = RunningMeanStd()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Computes GAE with improved stability features.

        Args:
            tensordict (TensorDictBase): Input dictionary containing rewards, values, etc.

        Returns:
            TensorDictBase: Updated dictionary with:
                - "advantage": normalized advantage
                - "value_target": clamped value target (if present)
        """
        # ============ REWARD PREPROCESSING ============
        reward_key = ("next", "reward")
        if reward_key in tensordict.keys(include_nested=True):
            rewards = tensordict.get(reward_key)

            # Update running statistics
            if self.normalize_rewards:
                self.reward_stats.update(rewards.detach())

            # Clip rewards if configured
            if self.clip_rewards > 0:
                rewards = rewards.clamp(-self.clip_rewards, self.clip_rewards)

            # Normalize rewards using running statistics
            if self.normalize_rewards and self.reward_stats.count > 100:
                # Only normalize after collecting enough samples
                rewards = self.reward_stats.normalize(rewards)

            tensordict.set(reward_key, rewards)

        # ============ COMPUTE GAE ============
        tensordict = super().forward(tensordict)

        # ============ ADVANTAGE NORMALIZATION ============
        adv = tensordict.get("advantage")
        if adv is not None and adv.numel() > 1:
            # Robust normalization with outlier handling
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False)

            # Prevent division by very small std (indicates all similar advantages)
            if adv_std > 1e-6:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            else:
                # If std is too small, just center the advantages
                adv = adv - adv_mean

            # Clip extreme advantages to prevent large policy updates
            adv = adv.clamp(-10.0, 10.0)
            tensordict.set("advantage", adv)

        # ============ VALUE TARGET CLAMPING ============
        if "value_target" in tensordict.keys():
            vt = tensordict.get("value_target")

            # Update return statistics for monitoring
            self.return_stats.update(vt.detach())

            # Clamp to prevent extreme targets
            vt = vt.clamp(-self.value_clip_range, self.value_clip_range)
            tensordict.set("value_target", vt)

        return tensordict

    def get_stats(self):
        """Return current normalization statistics for logging."""
        return {
            "reward_mean": self.reward_stats.mean.item() if hasattr(self.reward_stats.mean, 'item') else self.reward_stats.mean,
            "reward_std": torch.sqrt(self.reward_stats.var).item() if hasattr(self.reward_stats.var, 'item') else self.reward_stats.var ** 0.5,
            "return_mean": self.return_stats.mean.item() if hasattr(self.return_stats.mean, 'item') else self.return_stats.mean,
            "return_std": torch.sqrt(self.return_stats.var).item() if hasattr(self.return_stats.var, 'item') else self.return_stats.var ** 0.5,
        }
