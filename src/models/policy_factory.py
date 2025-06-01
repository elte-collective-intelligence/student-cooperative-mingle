import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import (
    ProbabilisticActor,
    NormalParamExtractor,
    TanhNormal,
    MultiAgentMLP
)

def build_policy(env, policy_config: dict, device: torch.device) -> ProbabilisticActor:
    """
    Builds a ProbabilisticActor policy using configuration dict.

    Args:
        env: Environment with observation/action specs.
        device: Torch device to place the model.
        policy_config: Dictionary from YAML config with 'policy' and 'mlp' settings.

    Returns:
        A configured ProbabilisticActor.
    """

    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }
    
    mlp = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1],
        n_agents=env.n_agents,
        centralised=policy_config["centralised"],
        share_params=policy_config["share_params"],
        device=device,
        depth=policy_config["depth"],
        num_cells=policy_config["num_cells"],
        activation_class=activation_map[policy_config["activation"].lower()],
    )

    class MinScaleModule(nn.Module):
        def __init__(self, min_scale: float):
            super().__init__()
            self.min_scale = min_scale
            self.param_extractor = NormalParamExtractor()

        def forward(self, x):
            loc, scale = self.param_extractor(x)
            scale = torch.nn.functional.softplus(scale) + self.min_scale
            return loc, scale

    combined_module = nn.Sequential(mlp, MinScaleModule(policy_config["min_scale"]))

    policy_module = TensorDictModule(
        module=combined_module,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    return ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=["loc", "scale"],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True
    )
