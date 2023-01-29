from typing import Tuple, Dict, Union, List

import gym
import networkx as nx
import torch as th
from torch import nn

from heuristic_layers import P2CLoadBalanceHeuristic, HADRLHeuristic


class HADRLActor(nn.Module):
    """ Actor network for the HA-DRL [1] algorithm

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            action_space: gym.Space,
            psn: nx.Graph,
            net_arch: Union[List[int], Dict[str, List[int]]],
            servers_map_idx_id: Dict[int, int],
            in_features: int,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
    ):
        """ Constructor

        :param action_space: action space
        :param psn: env's physical substrate network
        :param servers_map_idx_id: map (dict) between servers indexes (agent's actions) and their ids
        :param use_heuristic: if True, actor will use P2C heuristic
        """
        super().__init__()
        self.use_heuristic = use_heuristic
        heu_class = heu_kwargs.get('heu_class', HADRLHeuristic)

        # layers
        dims = [in_features] + net_arch['pi']
        modules = nn.ModuleList()
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.Tanh())

        if self.use_heuristic:
            self.heu_layer = heu_class(action_space, servers_map_idx_id, psn,
                                  **heu_kwargs).requires_grad_(False)

        self.layers = nn.Sequential(*modules)

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        x = self.layers(x)
        if self.use_heuristic:
            x = self.heu_layer(x, obs)
        return x


class HADRLCritic(nn.Module):
    """ Critic network for the HA-DRL [1] algorithm

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            in_features: int,
            net_arch: List[Union[int, Dict[str, List[int]]]]
    ):
        """ Constructor

        :param in_features: number of features extracted by the features extractor,
            i.e., input dim of the first layer of the network
        """
        super().__init__()
        dims = [in_features] + net_arch['vf']
        modules = nn.ModuleList()
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU())
        self.layers = nn.Sequential(*modules)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


class HADRLActorCriticNet(nn.Module):
    """
    Actor-Critic network for the HA-DRL [1] algorithm

    [1] https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            action_space: gym.Space,
            psn: nx.Graph,
            net_arch: List[Union[int, Dict[str, List[int]]]],
            servers_map_idx_id: Dict[int, int],
            features_dim: Union[int, Dict[str, int]],
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
    ):
        """ Constructor

        :param action_space: action space
        :param psn: env's physical substrate network
        :param servers_map_idx_id: map (dict) between servers indexes (agent's actions) and their ids
        :param policy_features_dim:
        :param value_features_dim:
        :param gcn_out_channels: number of output channels of the GCN layer
        :param nspr_out_features: output dim of the layer that receives the NSPR state
        :param use_heuristic: if True, actor will use P2C heuristic
        """
        super(HADRLActorCriticNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]

        if isinstance(features_dim, int):
            policy_features_dim = value_features_dim = features_dim
        else:
            policy_features_dim = features_dim['pi']
            value_features_dim = features_dim['vf']

        # policy network
        self.policy_net = HADRLActor(action_space, psn, net_arch,
                                     servers_map_idx_id, policy_features_dim,
                                     use_heuristic, heu_kwargs)
        # value network
        self.value_net = HADRLCritic(value_features_dim, net_arch)

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
