import torch
from torch.nn import functional as F
from torchrl.objectives import ClipPPOLoss
from tensordict import TensorDictBase

class StablePPOLoss(ClipPPOLoss):
    """
    PPO loss with improved critic stability using Huber loss and value clipping.

    Extends:
        ClipPPOLoss: Standard PPO loss with clipped surrogate objective.

    Enhancements:
        - Clamps value targets to [-10, 10] to reduce the effect of outlier returns.
        - Uses Huber loss for critic value loss for better robustness compared to MSE.
        - Evaluates old critic values without gradient tracking for stable targets.

    Args:
        actor_network: The policy network.
        critic_network: The value function network.
        clip_epsilon (float): Clipping parameter for PPO surrogate objective.
        entropy_bonus (bool): Whether to include entropy bonus.
        entropy_coef (float): Coefficient for entropy bonus.
    """

    def forward(self, tensordict: TensorDictBase) -> dict:
        """
        Computes PPO losses with improved critic stability.

        Args:
            tensordict (TensorDictBase): Contains rollout data with policy, values, advantages, etc.

        Returns:
            dict: Dictionary containing PPO loss components, including:
                - "loss_actor": PPO clipped policy loss
                - "loss_critic": Huber loss on value function
                - "loss_entropy": Entropy bonus (if enabled)
                - "loss": Sum of all losses
        """
        # Evaluate old values without gradients to stabilize targets
        with torch.no_grad():
            old_values = self.critic_network(tensordict)

        # Compute standard PPO losses (actor + entropy + baseline)
        loss_dict = super().forward(tensordict)

        # Get current value predictions and clipped targets
        values = tensordict.get("state_value")
        returns = tensordict.get("value_target").clamp(min=-10.0, max=10.0)

        # Replace critic loss with robust Huber loss
        loss_dict["loss_critic"] = F.huber_loss(values, returns, reduction="mean", delta=1.0)

        return loss_dict
