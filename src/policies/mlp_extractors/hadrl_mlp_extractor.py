from typing import Tuple, Dict

import gym
import networkx as nx
import torch as th
from torch import nn

from heuristic_layers import P2CLoadBalanceHeuristic


class HADRLActor(nn.Module):
    """ Actor network for the HA-DRL [1] algorithm

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            action_space: gym.Space,
            psn: nx.Graph,
            servers_map_idx_id: Dict[int, int],
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
    ):
        """ Constructor

        :param action_space: action space
        :param psn: env's physical substrate network
        :param servers_map_idx_id: map (dict) between servers indexes (agent's actions) and their ids
        :param gcn_out_channels: number of output channels of the GCN layer
        :param nspr_out_features: output dim of the layer that receives the NSPR state
        :param use_heuristic: if True, actor will use P2C heuristic
        """
        super().__init__()
        n_nodes = len(psn.nodes)
        self.use_heuristic = use_heuristic
        # layers
        self.linear = nn.Linear(
            in_features=n_nodes * gcn_out_channels + nspr_out_features,
            out_features=n_nodes)
        self.heuristic = P2CLoadBalanceHeuristic(
            action_space, servers_map_idx_id, psn, **heu_kwargs).requires_grad_(False)

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        x = th.tanh(self.linear(x))
        if self.use_heuristic:
            x = self.heuristic(x, obs)
        x = th.softmax(x, dim=1)
        return x


class HSDRLCritic(nn.Module):
    """ Critic network for the HA-DRL [1] algorithm

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            psn: nx.Graph,
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4
    ):
        """ Constructor

        :param psn: env's physical substrate network
        :param gcn_out_channels: number of output channels of the GCN layer
        :param nspr_out_features: output dim of the layer that receives the NSPR state
        """
        super().__init__()
        n_nodes = len(psn.nodes)
        self.final_fcs = nn.Sequential(
            nn.Linear(
                in_features=n_nodes * gcn_out_channels + nspr_out_features,
                out_features=n_nodes),
            nn.ReLU(),
            nn.Linear(in_features=n_nodes, out_features=1),
            nn.ReLU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.final_fcs(x)


class HADRLActorCriticNet(nn.Module):
    """
    Actor-Critic network for the HA-DRL [1] algorithm

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            action_space: gym.Space,
            psn: nx.Graph,
            servers_map_idx_id: Dict[int, int],
            feature_dim: int,
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
    ):
        """ Constructor

        :param action_space: action space
        :param psn: env's physical substrate network
        :param servers_map_idx_id: map (dict) between servers indexes (agent's actions) and their ids
        :param feature_dim:
        :param gcn_out_channels: number of output channels of the GCN layer
        :param nspr_out_features: output dim of the layer that receives the NSPR state
        :param use_heuristic: if True, actor will use P2C heuristic
        """
        super(HADRLActorCriticNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = len(psn.nodes)
        self.latent_dim_vf = 1
        # policy network
        self.policy_net = HADRLActor(action_space, psn, servers_map_idx_id,
                                     gcn_out_channels, nspr_out_features,
                                     use_heuristic, heu_kwargs)
        # value network
        self.value_net = HSDRLCritic(psn, gcn_out_channels, nspr_out_features)

    def forward(self, features: th.Tensor, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features, obs), self.value_net(features)

    def forward_actor(self, features: th.Tensor, obs: th.Tensor) -> th.Tensor:
        return self.policy_net(features, obs)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
