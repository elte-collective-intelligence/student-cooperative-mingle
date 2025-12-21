import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    ProbabilisticActor,
    NormalParamExtractor,
    TanhNormal,
    MultiAgentMLP
)


def build_continuous_comm_policy(env, policy_config: dict, device: torch.device) -> TensorDictSequential:
    """
    Builds a policy that outputs both movement actions and continuous embeddings.

    Args:
        env: Environment with observation/action specs (must have embedding action spec)
        policy_config: Dictionary with 'policy' and 'mlp' settings
        device: Torch device

    Returns:
        TensorDictSequential policy module
    """
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }

    # Shared encoder
    n_agent_inputs = env.observation_spec["observation"].shape[-1]
    n_agents = env.n_agents
    hidden_dim = policy_config["num_cells"]

    encoder = MultiAgentMLP(
        n_agent_inputs=n_agent_inputs,
        n_agent_outputs=hidden_dim,
        n_agents=n_agents,
        centralised=policy_config["centralised"],
        share_params=policy_config["share_params"],
        device=device,
        depth=policy_config["depth"],
        num_cells=hidden_dim,
        activation_class=activation_map[policy_config["activation"].lower()],
    )

    # Movement head (continuous actions)
    movement_head = MultiAgentMLP(
        n_agent_inputs=hidden_dim,
        n_agent_outputs=2 * 2,  # 2 actions * 2 (loc, scale)
        n_agents=n_agents,
        centralised=False,
        share_params=policy_config["share_params"],
        device=device,
        depth=1,
        num_cells=hidden_dim,
        activation_class=activation_map[policy_config["activation"].lower()],
    )

    # Embedding head (continuous communication)
    embedding_dim = env.action_spec["embedding"].shape[-1]
    embedding_head = MultiAgentMLP(
        n_agent_inputs=hidden_dim,
        n_agent_outputs=embedding_dim,
        n_agents=n_agents,
        centralised=False,
        share_params=policy_config["share_params"],
        device=device,
        depth=1,
        num_cells=hidden_dim // 2,
        activation_class=activation_map[policy_config["activation"].lower()],
    )

    # Combined network
    class EmbeddingCommunicativeNetwork(nn.Module):
        def __init__(self, encoder, movement_head, embedding_head, min_scale):
            super().__init__()
            self.encoder = encoder
            self.movement_head = movement_head
            self.embedding_head = embedding_head
            self.min_scale = min_scale
            self.param_extractor = NormalParamExtractor()

        def forward(self, obs):
            features = self.encoder(obs)

            # Movement outputs
            movement_params = self.movement_head(features)
            loc, scale = self.param_extractor(movement_params)
            scale = torch.nn.functional.softplus(scale) + self.min_scale

            # Embedding output (tanh to bound to [-1, 1])
            embedding = torch.tanh(self.embedding_head(features))

            return loc, scale, embedding

    combined_net = EmbeddingCommunicativeNetwork(
        encoder, movement_head, embedding_head, policy_config["min_scale"]
    )

    # Wrap in TensorDictModule
    policy_module = TensorDictModule(
        module=combined_net,
        in_keys=["observation"],
        out_keys=["loc", "scale", "embedding"]
    )

    # Movement distribution
    movement_actor = ProbabilisticActor(
        module=TensorDictModule(
            lambda loc, scale: (loc, scale),
            in_keys=["loc", "scale"],
            out_keys=["loc", "scale"]
        ),
        spec=env.action_spec["action"],
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec["action"].space.low,
            "high": env.action_spec["action"].space.high,
        },
        return_log_prob=True
    )

    # Embedding pass-through (deterministic, no distribution needed)
    # Just copy embedding to output (already bounded by tanh)

    # Combine into sequential module
    return TensorDictSequential(
        policy_module,
        movement_actor
    )
