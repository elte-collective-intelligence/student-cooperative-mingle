"""
Custom PPO Loss for multi-action policies (movement + communication).

This extends the standard ClipPPOLoss to handle policies that output
multiple action types with separate distributions.

FIXES APPLIED:
1. Use Huber loss instead of MSE for critic stability
2. Add proper separation of PG term and entropy term for logging
3. Clamp value targets to prevent extreme values
4. Return only actual loss terms (not diagnostics) to prevent training loop issues
5. Add entropy annealing support
6. KL-based early stopping with proper threshold (0.03-0.05)
7. Learning rate scaling when KL is high
"""

import torch
import torch.nn.functional as F
from torchrl.objectives import ClipPPOLoss
from tensordict import TensorDict


class MultiActionPPOLoss(ClipPPOLoss):
    """
    PPO Loss that handles multiple action types.

    This loss combines the PPO objective for both movement actions
    and discrete message actions.

    Improvements over standard ClipPPOLoss:
    - Uses Huber loss for critic (more stable with outliers)
    - Clamps value targets to prevent extreme gradients
    - Properly separates PG term and entropy for logging
    - Supports entropy annealing
    - KL-based early stopping with threshold 0.03-0.05
    """

    def __init__(
        self,
        actor_network,
        critic_network,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coeff=0.5,  # Standard value function coefficient
        normalize_advantage=True,
        value_clip_range=(-100.0, 100.0),
        kl_threshold=0.04,  # Target KL threshold for early stopping
        **kwargs
    ):
        # Initialize parent with basic settings
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            clip_epsilon=clip_epsilon,
            entropy_bonus=entropy_bonus,
            entropy_coef=entropy_coef,
            critic_coeff=critic_coeff,
            normalize_advantage=normalize_advantage,
            **kwargs
        )
        self.value_clip_range = value_clip_range
        self._current_entropy_coef = entropy_coef
        self._initial_entropy_coef = entropy_coef
        self.kl_threshold = kl_threshold
        self._lr_scale = 1.0  # For dynamic LR scaling

        # Diagnostics storage (not part of loss)
        self.last_diagnostics = {}

    def set_entropy_coef(self, coef: float):
        """Update entropy coefficient (for annealing)."""
        self._current_entropy_coef = coef

    def get_lr_scale(self) -> float:
        """Get learning rate scale based on KL divergence."""
        return self._lr_scale

    def should_stop_early(self) -> bool:
        """Check if KL exceeds threshold for early stopping."""
        if 'kl_approx' in self.last_diagnostics:
            kl = self.last_diagnostics['kl_approx']
            if kl > self.kl_threshold * 1.5:
                return True
            # Scale down LR if KL is moderately high
            if kl > self.kl_threshold:
                self._lr_scale = 0.5
            else:
                self._lr_scale = 1.0
        return False

    def _log_weight(self, tensordict: TensorDict):
        """
        Override to handle multiple log probabilities.

        Uses the combined log_prob from CombinedLogProbWrapper.
        The wrapper already combines movement + message log_probs into "action_log_prob".
        """
        # Get old COMBINED log prob (stored during data collection by wrapper)
        # The wrapper stores combined (movement + message) in "action_log_prob"
        old_log_prob = tensordict.get("action_log_prob")

        if old_log_prob is None:
            raise ValueError("No action_log_prob found in tensordict. "
                           "Make sure to use CombinedLogProbWrapper.")

        # IMPORTANT: Clone and detach OLD log probs
        old_log_prob = old_log_prob.detach().clone()

        # Compute new log probs by running policy (wrapper combines them)
        with torch.set_grad_enabled(True):
            td_out = self.actor_network(tensordict)

        # Get new COMBINED log prob (wrapper already combines movement + message)
        current_log_prob = td_out.get("action_log_prob")

        if current_log_prob is None:
            raise ValueError("No action_log_prob in policy output. "
                           "Make sure CombinedLogProbWrapper is wrapping the policy.")

        # Compute log weight (importance ratio in log space)
        log_weight = current_log_prob - old_log_prob

        # Compute KL divergence approximation (using formula: KL ≈ (old - new) log probs)
        kl_approx = (old_log_prob - current_log_prob).detach()

        return log_weight, current_log_prob, kl_approx

    def _get_policy_entropy(self, tensordict: TensorDict):
        """
        Override to compute entropy for both action types.
        """
        # Run policy to get distributions
        with torch.set_grad_enabled(True):
            td_out = self.actor_network(tensordict)

        entropy_list = []

        # Movement entropy (if available)
        if "scale" in td_out:
            # For TanhNormal distribution, approximate entropy using Gaussian entropy
            # H = 0.5 * log(2 * pi * e * sigma^2) for each dimension
            scale = td_out["scale"]
            movement_entropy = 0.5 * torch.log(2 * torch.pi * torch.e * scale.pow(2) + 1e-8)
            movement_entropy = movement_entropy.sum(dim=-1)  # Sum over action dimensions
            entropy_list.append(movement_entropy)

        # Message entropy (if available)
        if "logits" in td_out:
            # For categorical distribution, compute entropy
            logits = td_out["logits"]
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            message_entropy = -(probs * log_probs).sum(dim=-1)
            entropy_list.append(message_entropy)

        if entropy_list:
            # Sum entropies (not average - we want total entropy)
            total_entropy = sum(entropy_list)
            # Flatten to match batch
            if total_entropy.dim() > 1:
                total_entropy = total_entropy.flatten(0, -1)
            return total_entropy
        else:
            return torch.tensor(0.0, device=tensordict.device)

    def forward(self, tensordict: TensorDict):
        """
        Compute PPO loss with multiple actions.

        Returns ONLY the loss terms that should be optimized.
        Diagnostics are stored separately in self.last_diagnostics.
        """
        # Get advantage and value target
        advantage = tensordict.get(self.tensor_keys.advantage)

        # Compute log weight (importance ratio)
        log_weight, current_log_prob, kl = self._log_weight(tensordict)

        # Ensure advantage and log_weight have compatible shapes
        if advantage.dim() > 1:
            advantage = advantage.flatten()
        if log_weight.dim() > 1:
            log_weight = log_weight.flatten()

        # Match shapes if needed
        if advantage.shape != log_weight.shape:
            min_len = min(len(advantage), len(log_weight))
            advantage = advantage[:min_len]
            log_weight = log_weight[:min_len]

        # Clamp log_weight to prevent numerical instability
        log_weight = log_weight.clamp(-10.0, 10.0)

        # ============ CORRECT CLIPPED PPO LOSS ============
        # ratio = exp(new_logp - old_logp)
        ratio = torch.exp(log_weight)

        # Unclipped objective
        unclipped = ratio * advantage

        # Clipped objective
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        ) * advantage

        # PPO uses min to be conservative
        # - Positive advantage: min prevents ratio from being too high
        # - Negative advantage: min prevents ratio from being too low
        gain = torch.min(unclipped, clipped)

        # Policy gradient loss (negative because we MINIMIZE loss to MAXIMIZE gain)
        pg_loss = -gain.mean()

        # ============ CRITIC LOSS (Huber for stability) ============
        value_pred = self.critic_network(tensordict).get("state_value")
        value_target = tensordict.get(self.tensor_keys.value_target)

        # Flatten if needed
        if value_pred.dim() > 1:
            value_pred = value_pred.flatten()
        if value_target.dim() > 1:
            value_target = value_target.flatten()

        # Match shapes
        if value_pred.shape != value_target.shape:
            min_len = min(len(value_pred), len(value_target))
            value_pred = value_pred[:min_len]
            value_target = value_target[:min_len]

        # Clamp value targets to prevent extreme gradients
        value_target_clamped = value_target.clamp(
            self.value_clip_range[0], self.value_clip_range[1]
        ).detach()  # IMPORTANT: detach targets!

        # Use Huber loss for stability (delta=1.0)
        critic_loss = F.huber_loss(value_pred, value_target_clamped, delta=1.0)

        # ============ ENTROPY BONUS ============
        if self.entropy_bonus:
            entropy = self._get_policy_entropy(tensordict)
            entropy_mean = entropy.mean()
            # Entropy bonus: we WANT high entropy, so we SUBTRACT it from loss
            entropy_loss = -entropy_mean * self._current_entropy_coef
        else:
            entropy_mean = torch.tensor(0.0, device=tensordict.device)
            entropy_loss = torch.tensor(0.0, device=tensordict.device)

        # ============ STORE DIAGNOSTICS ============
        with torch.no_grad():
            # Clip fraction (how often clipping was active)
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

            self.last_diagnostics = {
                "pg_loss": pg_loss.item(),
                "entropy": entropy_mean.item(),
                "entropy_loss": entropy_loss.item(),
                "critic_loss": critic_loss.item(),
                "ratio_mean": ratio.mean().item(),
                "ratio_std": ratio.std().item(),
                "clip_fraction": clip_fraction,
                "advantage_mean": advantage.mean().item(),
                "advantage_std": advantage.std().item(),
                "value_pred_mean": value_pred.mean().item(),
                "value_target_mean": value_target.mean().item(),
                "kl_approx": kl.mean().item() if kl is not None else 0.0,
            }

        # ============ RETURN ONLY LOSS TERMS ============
        return TensorDict({
            "loss_objective": pg_loss,
            "loss_critic": critic_loss * self.critic_coeff,
            "loss_entropy": entropy_loss,
        }, batch_size=[])
