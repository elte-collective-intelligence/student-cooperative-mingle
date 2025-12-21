"""
Policy wrapper that combines multiple action log probs for standard PPO.

This allows using standard ClipPPOLoss with multi-action policies.
"""
import torch
from tensordict.nn import TensorDictModule


class CombinedLogProbWrapper(TensorDictModule):
    """
    Wraps a multi-action policy to combine log probs into single key.

    This enables using standard ClipPPOLoss which expects:
    - "sample_log_prob" as the OLD log_prob (stored during data collection)
    - The actor to recompute log_probs during training

    Our policy outputs:
    - "sample_log_prob_movement" for continuous movement actions
    - "sample_log_prob_message" for discrete message actions

    We combine them into "sample_log_prob" for ClipPPOLoss.
    """

    def __init__(self, policy):
        """
        Args:
            policy: The underlying multi-action policy (TensorDictSequential)
        """
        super().__init__(
            module=policy,
            in_keys=policy.in_keys,
            out_keys=policy.out_keys
        )
        self.policy = policy

    def forward(self, tensordict):
        """
        Run policy and combine log probs.
        """
        # Run the underlying policy
        td_out = self.policy(tensordict)

        # Get both log probs (using explicit keys we set in discrete_comm_policy)
        movement_log_prob = td_out.get("sample_log_prob_movement", None)
        message_log_prob = td_out.get("sample_log_prob_message", None)

        # Combine them (sum in log space = product in probability space)
        if movement_log_prob is not None and message_log_prob is not None:
            combined_log_prob = movement_log_prob + message_log_prob

            # Store as "action_log_prob" - this is what ClipPPOLoss expects by default
            # (despite the name suggesting it's the "action" log_prob, it's used for both
            # old and new log_probs in PPO - the loss module handles the comparison)
            td_out.set("action_log_prob", combined_log_prob)

            # Also set sample_log_prob for compatibility
            td_out.set("sample_log_prob", combined_log_prob)

            # Keep the individual ones for debugging
            td_out.set("movement_log_prob_debug", movement_log_prob)
            td_out.set("message_log_prob_debug", message_log_prob)

        return td_out
