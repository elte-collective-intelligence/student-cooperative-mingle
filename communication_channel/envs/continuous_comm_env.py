import torch
import math
from typing import List, Optional, Tuple
from tensordict import TensorDict
from torchrl.data import Composite, Bounded, Unbounded, Categorical

from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import RewardModule


class MingleEnvWithEmbeddings(MingleEnv):
    """
    MingleEnv extended with continuous learned embedding communication.

    Agents can broadcast continuous embedding vectors that can encode
    rich information learned during training.

    Parameters:
        embedding_dim (int): Dimension of communication embeddings (default: 16)
        comm_range (float, optional): Communication range in distance units.
                                       If None, messages are aggregated globally.
        aggregation (str): How to aggregate embeddings ("mean", "attention")
        All other parameters inherited from MingleEnv
    """

    def __init__(
        self,
        n_agents: int = 2,
        n_rooms: int = 2,
        arena_radius: float = 10.0,
        center_radius: float = 3.0,
        max_steps: int = 300,
        spinning_phase_range: Tuple[int, int] = (50, 100),
        room_radius: float = 3.0,
        room_capacity: int = 2,
        reward_modules: Optional[List[RewardModule]] = None,
        reward_managers: Optional[dict] = None,
        phase_mode: str = "both",
        embedding_dim: int = 16,
        comm_range: Optional[float] = None,
        aggregation: str = "mean",
    ) -> None:
        # Initialize parent first
        super().__init__(
            n_agents=n_agents,
            n_rooms=n_rooms,
            arena_radius=arena_radius,
            center_radius=center_radius,
            max_steps=max_steps,
            spinning_phase_range=spinning_phase_range,
            room_radius=room_radius,
            room_capacity=room_capacity,
            reward_modules=reward_modules,
            reward_managers=reward_managers,
            phase_mode=phase_mode
        )

        self.embedding_dim = embedding_dim
        self.comm_range = comm_range
        self.aggregation = aggregation
        self.embedding_buffer = torch.zeros(
            self.n_agents, self.embedding_dim, device=self.device
        )

        # Calculate observation dimension:
        # Base obs: 14 + aggregated embedding: embedding_dim
        new_obs_dim = 14 + self.embedding_dim

        # Update observation spec
        self.observation_spec = Composite({
            "observation": Unbounded(shape=(self.n_agents, new_obs_dim), device=self.device),
        })

        # Update action spec to include embedding output
        self.action_spec = Composite({
            "action": Bounded(
                low=-self.max_speed,
                high=self.max_speed,
                shape=(self.n_agents, 2),
                device=self.device,
            ),
            "embedding": Bounded(
                low=-1.0,
                high=1.0,
                shape=(self.n_agents, self.embedding_dim),
                device=self.device,
            ),
        })

    def _add_embeddings_to_obs(self, base_obs: torch.Tensor) -> torch.Tensor:
        """
        Add aggregated communication embeddings to observations.

        Args:
            base_obs: Base observation tensor from parent class (n_agents, 14)

        Returns:
            Extended observation with embeddings (n_agents, 14 + embedding_dim)
        """
        n_agents = base_obs.shape[0]

        if self.comm_range is None:
            # Global aggregation
            if self.aggregation == "mean":
                embeddings = self._aggregate_global_mean(n_agents)
            elif self.aggregation == "attention":
                embeddings = self._aggregate_global_attention(n_agents)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        else:
            # Local aggregation
            if self.aggregation == "mean":
                embeddings = self._aggregate_local_mean(n_agents)
            elif self.aggregation == "attention":
                embeddings = self._aggregate_local_attention(n_agents)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        # Concatenate with base observations
        return torch.cat([base_obs, embeddings], dim=-1)

    def _aggregate_global_mean(self, n_agents: int) -> torch.Tensor:
        """
        Aggregate embeddings globally using mean pooling.

        Returns:
            Tensor of shape (n_agents, embedding_dim)
        """
        # Sum of all embeddings
        embeddings_sum = self.embedding_buffer.sum(dim=0, keepdim=True)  # (1, embedding_dim)
        embeddings_sum = embeddings_sum.repeat(n_agents, 1)  # (n_agents, embedding_dim)

        # Subtract self-embedding for each agent
        neighbor_embeddings = (embeddings_sum - self.embedding_buffer) / max(1, n_agents - 1)

        return neighbor_embeddings

    def _aggregate_global_attention(self, n_agents: int) -> torch.Tensor:
        """
        Aggregate embeddings globally using attention mechanism.

        Returns:
            Tensor of shape (n_agents, embedding_dim)
        """
        # Compute attention scores based on embedding similarity
        # scores[i, j] = similarity between agent i and agent j's embedding
        scores = torch.matmul(
            self.embedding_buffer, self.embedding_buffer.t()
        )  # (n_agents, n_agents)

        # Mask out self-attention
        mask = torch.eye(n_agents, device=self.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (n_agents, n_agents)

        # Aggregate embeddings using attention
        aggregated = torch.matmul(attention_weights, self.embedding_buffer)  # (n_agents, embedding_dim)

        return aggregated

    def _aggregate_local_mean(self, n_agents: int) -> torch.Tensor:
        """
        Aggregate embeddings from nearby agents using mean pooling.

        Returns:
            Tensor of shape (n_agents, embedding_dim)
        """
        positions = self.agent_positions
        distances = torch.cdist(positions, positions)  # (n_agents, n_agents)

        # Create mask for agents within comm_range (excluding self)
        comm_mask = (distances <= self.comm_range) & (distances > 0)

        aggregated = []
        for i in range(n_agents):
            if comm_mask[i].any():
                # Mean of nearby agents' embeddings
                neighbor_embeddings = self.embedding_buffer[comm_mask[i]]
                aggregated.append(neighbor_embeddings.mean(dim=0))
            else:
                # No neighbors: return zero vector
                aggregated.append(torch.zeros(self.embedding_dim, device=self.device))

        return torch.stack(aggregated)  # (n_agents, embedding_dim)

    def _aggregate_local_attention(self, n_agents: int) -> torch.Tensor:
        """
        Aggregate embeddings from nearby agents using distance-based attention.

        Returns:
            Tensor of shape (n_agents, embedding_dim)
        """
        positions = self.agent_positions
        distances = torch.cdist(positions, positions)  # (n_agents, n_agents)

        # Create attention weights based on distance
        comm_mask = (distances <= self.comm_range) & (distances > 0)
        attention_weights = torch.exp(-distances / self.comm_range)
        attention_weights = attention_weights * comm_mask.float()

        # Normalize attention weights
        attention_weights = attention_weights / (
            attention_weights.sum(dim=1, keepdim=True) + 1e-8
        )

        # Weighted sum of neighbor embeddings
        aggregated = torch.matmul(attention_weights, self.embedding_buffer)

        return aggregated  # (n_agents, embedding_dim)

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset environment and initialize embedding buffer."""
        # Call parent reset
        td = super()._reset(tensordict)

        # Initialize embedding buffer with zeros
        self.embedding_buffer = torch.zeros(
            self.n_agents, self.embedding_dim, device=self.device
        )

        # Add embeddings to observations
        base_obs = td["observation"]
        extended_obs = self._add_embeddings_to_obs(base_obs)
        td["observation"] = extended_obs

        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Step environment with actions and embeddings.

        Expected tensordict keys:
            - "action": (n_agents, 2) movement actions
            - "embedding": (n_agents, embedding_dim) communication embeddings
        """
        # Extract and store embeddings before stepping
        if "embedding" in tensordict:
            self.embedding_buffer = tensordict["embedding"].clone()

        # Call parent step with movement actions only
        movement_td = TensorDict({
            "action": tensordict["action"]
        }, batch_size=tensordict.batch_size)

        td = super()._step(movement_td)

        # Add embeddings to next observation
        base_obs = td["observation"]
        extended_obs = self._add_embeddings_to_obs(base_obs)
        td["observation"] = extended_obs

        return td
