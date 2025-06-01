import torch
from torch import nn
from torchrl.modules import ValueOperator
from torchrl.modules import (
    MultiAgentMLP
)

def build_critic(env, critic_config: dict, device: torch.device = torch.device("cpu")) -> ValueOperator:
    """
    Builds a critic (ValueOperator) for multi-agent value estimation.

    Args:
        env: The environment instance (used to get input/output shapes).
        critic_config (dict): Configuration for the critic MLP.
        device (torch.device): Device to place the model on.

    Returns:
        ValueOperator: TorchRL-compatible value function.
    """
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1],
        n_agent_outputs=1,  # Single value per agent
        n_agents=env.n_agents,
        centralised=critic_config.get("centralised", False),
        share_params=critic_config.get("share_params", False),
        device=device,
        depth=critic_config.get("depth", 3),
        num_cells=critic_config.get("num_cells", 256),
        activation_class=activation_map[critic_config["activation"].lower()],
    )

    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"],
    )
    return critic
