import torch
import torch.nn as nn
from torchrl.modules import MLP, NormalParamExtractor
from tensordict.nn import TensorDictModule


class MAPPOActor(nn.Module):
    """
    Shared MAPPO actor that outputs (loc, scale) for Gaussian actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        depth: int = 2,
        min_scale: float = 1e-4,
    ):
        super().__init__()

        layers = [hidden_size] * depth

        # Shared MLP
        self.mlp = MLP(
            in_features=obs_dim,
            out_features=action_dim * 2,   # loc + scale
            num_cells=layers,
            activation_class=nn.ReLU,
        )

        # TorchRL expects scale_mapping to be a STRING
        self.extractor = NormalParamExtractor(
            scale_mapping="softplus"
        )

        self.min_scale = min_scale

        # Wrap with TensorDictModule
        self.module = TensorDictModule(
            nn.Sequential(self.mlp, self.extractor),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

    def forward(self, x):
        td = self.module(x)
        loc = td["loc"]
        scale = td["scale"].clamp(min=self.min_scale)
        return loc, scale
