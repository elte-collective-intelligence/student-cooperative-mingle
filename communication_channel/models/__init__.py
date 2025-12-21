"""Communication-enabled policy networks"""

from communication_channel.models.discrete_comm_policy import build_discrete_comm_policy
from communication_channel.models.continuous_comm_policy import build_continuous_comm_policy

__all__ = ["build_discrete_comm_policy", "build_continuous_comm_policy"]
