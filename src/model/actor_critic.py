import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def mlp(input_dim: int, hidden_dims=(128, 128), output_dim: int = None):
    """
    Simple MLP builder used by both actor and critic.
    """
    layers = []
    last_dim = input_dim

    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h

    if output_dim is not None:
        layers.append(nn.Linear(last_dim, output_dim))

    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """
    Shared actor network for MAPPO.

    Assumes:
      - observations are flattened to shape (batch, obs_dim)
      - discrete actions with action_dim possible actions
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims=(128, 128),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = mlp(obs_dim, hidden_dims, output_dim=action_dim)

    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over actions.

        obs: (batch, obs_dim)
        """
        # Make sure obs is 2D
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)

        logits = self.net(obs)
        return Categorical(logits=logits)

    def act(self, obs: torch.Tensor):
        """
        Convenience helper: sample actions + log_probs.
        """
        dist = self.forward(obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Used during training to compute log_probs and entropy for given actions.
        """
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class CentralValueNetwork(nn.Module):
    """
    Centralised critic for MAPPO.

    Assumes:
      - observations are flattened to shape (batch, obs_dim_total)
      - outputs state values of shape (batch, 1)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims=(128, 128),
    ):
        super().__init__()
        self.obs_dim = obs_dim

        self.net = mlp(obs_dim, hidden_dims, output_dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, obs_dim)
        returns: (batch, 1)
        """
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)

        value = self.net(obs)
        return value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Alias used by some trainers.
        """
        return self.forward(obs)
