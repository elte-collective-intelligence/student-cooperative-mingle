"""
Centralized Critic for Multi-Agent RL.

The critic takes global state (concatenated observations of all agents)
rather than individual agent observations, enabling better value estimation
for cooperative tasks.
"""

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP


def build_centralized_critic(env, critic_config: dict, device: torch.device):
    """
    Build a centralized critic that takes global state.

    For each agent, the critic sees the concatenated observations of ALL agents,
    allowing it to estimate the true state value in a cooperative setting.

    Args:
        env: Environment with observation spec
        critic_config: Configuration dictionary
        device: Torch device

    Returns:
        TensorDictModule for the centralized critic
    """
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }

    # Get observation dimensions
    obs_shape = env.observation_spec["observation"].shape
    n_agents = obs_shape[0]
    obs_dim = obs_shape[-1]

    # Global state dimension: all agents' observations concatenated
    global_state_dim = n_agents * obs_dim

    # Increased capacity for centralized critic
    hidden_dim = critic_config.get("num_cells", 128) * 2  # Wider network
    depth = critic_config.get("depth", 4)
    activation = critic_config.get("activation", "tanh").lower()

    class CentralizedCriticNetwork(nn.Module):
        """
        Centralized critic that processes global state.

        Input: Individual agent observation
        Process: Concatenate all agent observations to form global state
        Output: State value for each agent
        """

        def __init__(self):
            super().__init__()

            # Encoder: processes global state
            layers = []
            in_dim = global_state_dim

            for i in range(depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(activation_map[activation]())
                in_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)

            # Value head: outputs single value per agent
            self.value_head = nn.Linear(hidden_dim, n_agents)

        def forward(self, obs):
            """
            Forward pass for centralized critic.

            Args:
                obs: (batch_size, n_agents, obs_dim) or (n_agents, obs_dim)

            Returns:
                state_value: (batch_size, n_agents, 1) or (n_agents, 1)
            """
            # Handle different input shapes
            if obs.dim() == 2:
                # Single step: (n_agents, obs_dim)
                # Concatenate to get global state
                global_state = obs.flatten()  # (n_agents * obs_dim,)
                features = self.encoder(global_state)  # (hidden_dim,)
                values = self.value_head(features)  # (n_agents,)
                return values.unsqueeze(-1)  # (n_agents, 1)

            elif obs.dim() == 3:
                # Batch: (batch_size, n_agents, obs_dim)
                batch_size = obs.shape[0]
                # Concatenate all agents' obs for each batch
                global_state = obs.flatten(1, 2)  # (batch_size, n_agents * obs_dim)
                features = self.encoder(global_state)  # (batch_size, hidden_dim)
                values = self.value_head(features)  # (batch_size, n_agents)
                return values.unsqueeze(-1)  # (batch_size, n_agents, 1)

            else:
                raise ValueError(f"Unexpected observation shape: {obs.shape}")

    critic_net = CentralizedCriticNetwork().to(device)

    # Wrap in TensorDictModule
    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=["observation"],
        out_keys=["state_value"]
    )

    return critic_module


class SharedCritic(nn.Module):
    """
    Alternative: Shared critic with attention over other agents.

    This version uses attention to aggregate information from other agents
    rather than simple concatenation.
    """

    def __init__(self, obs_dim: int, n_agents: int, hidden_dim: int = 256):
        super().__init__()

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Individual encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Attention mechanism
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value_attn = nn.Linear(hidden_dim, hidden_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        """
        Forward with attention.

        Args:
            obs: (batch_size, n_agents, obs_dim) or (n_agents, obs_dim)

        Returns:
            values: (batch_size, n_agents, 1) or (n_agents, 1)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = obs.shape[0]

        # Encode all observations
        encoded = self.obs_encoder(obs)  # (batch, n_agents, hidden)

        # Compute attention
        Q = self.query(encoded)  # (batch, n_agents, hidden)
        K = self.key(encoded)    # (batch, n_agents, hidden)
        V = self.value_attn(encoded)  # (batch, n_agents, hidden)

        # Attention weights
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Aggregate
        context = torch.bmm(attn_weights, V)  # (batch, n_agents, hidden)

        # Combine with individual encoding
        combined = torch.cat([encoded, context], dim=-1)  # (batch, n_agents, hidden*2)

        # Output values
        values = self.value_head(combined)  # (batch, n_agents, 1)

        if squeeze_output:
            values = values.squeeze(0)

        return values
