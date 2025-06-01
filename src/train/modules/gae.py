from torchrl.objectives.value import GAE
from tensordict import TensorDictBase
import torch

class StableGAE(GAE):
    """
    A more stable variant of the Generalized Advantage Estimation (GAE) module.

    Improvements:
        1. Normalizes advantages to zero mean and unit variance — improves PPO stability.
        2. Clamps `value_target` to prevent extreme returns from destabilizing training.

    Parameters:
        gamma (float): Discount factor.
        lmbda (float): Lambda parameter for GAE.
        value_network (ValueOperator): The critic network to estimate state values.
        average_gae (bool): If True, averages advantages over all samples.
    """

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Computes and normalizes advantage and clamps value targets.

        Args:
            tensordict (TensorDictBase): Input dictionary containing rewards, values, etc.

        Returns:
            TensorDictBase: Updated dictionary with:
                - "advantage": normalized advantage
                - "value_target": clamped value target (if present)
        """
        # Compute advantages and value targets using base class method
        tensordict = super().forward(tensordict)

        # Normalize advantages: zero mean, unit variance
        adv = tensordict.get("advantage")
        if adv.numel() > 1:
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False) + 1e-8
            adv = (adv - adv_mean) / adv_std
            tensordict.set("advantage", adv)

        # Optionally clamp value targets to avoid extreme TD targets
        if "value_target" in tensordict:
            vt = tensordict.get("value_target")
            vt = vt.clamp(min=-10.0, max=10.0)
            tensordict.set("value_target", vt)

        return tensordict
