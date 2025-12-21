import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralizedCritic(nn.Module):
    """
    Centralized Value Function for MAPPO

    Input shape:
        (batch_size, n_agents, obs_dim)

    Output:
        (batch_size, 1)  → single global value for the whole team
    """

    def __init__(self, obs_dim: int, n_agents: int, hidden_dim: int = 128):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.input_dim = obs_dim * n_agents

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: shape (batch, n_agents, obs_dim)
        returns: shape (batch, 1)
        """

        if obs.dim() != 3:
            raise ValueError(
                f"Expected obs of shape (batch, n_agents, obs_dim), "
                f"got {obs.shape}"
            )

        # Flatten multi-agent observations
        x = obs.reshape(obs.shape[0], -1)  # (batch, n_agents * obs_dim)

        value = self.net(x)
        return value
