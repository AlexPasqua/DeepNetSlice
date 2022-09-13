from typing import Tuple

import gym
import networkx as nx
import torch as th
from torch import nn

from policy_nets import HADRLFeaturesExtractor


class HADRLActor(nn.Module):
    def __init__(self, observation_space: gym.Space, psn: nx.Graph, gcn_out_channels: int = 60,
                 nspr_out_features: int = 4):
        super().__init__()
        # self.features_extractor = HADRLFeaturesExtractor(observation_space, psn, nn.Tanh, gcn_out_channels, nspr_out_features)

        n_nodes = len(psn.nodes)
        self.final_fcs = nn.Sequential(
            nn.Linear(in_features=n_nodes * gcn_out_channels + nspr_out_features,
                      out_features=n_nodes),

            # TODO: put the heuristics here

            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = self.features_extractor(x)
        return self.final_fcs(x)


class HSDRLCritic(nn.Module):
    def __init__(self, observation_space: gym.Space, psn: nx.Graph, gcn_out_channels: int = 60,
                 nspr_out_features: int = 4):
        super().__init__()
        # self.features_extractor = HADRLFeaturesExtractor(observation_space, psn, nn.ReLU, gcn_out_channels, nspr_out_features)

        n_nodes = len(psn.nodes)
        self.final_fcs = nn.Sequential(
            nn.Linear(in_features=n_nodes * gcn_out_channels + nspr_out_features, out_features=n_nodes),
            nn.ReLU(),
            nn.Linear(in_features=n_nodes, out_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        # x = self.features_extractor(x)
        return self.final_fcs(x)


class HADRLActorCriticNet(nn.Module):
    def __init__(self, observation_space: gym.Space, psn: nx.Graph, feature_dim: int, gcn_out_channels: int = 60, nspr_out_features: int = 4):
        super(HADRLActorCriticNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = len(psn.nodes)
        self.latent_dim_vf = 1

        # policy network
        self.policy_net = HADRLActor(observation_space, psn, gcn_out_channels, nspr_out_features)

        # value network
        self.value_net = HSDRLCritic(observation_space, psn, gcn_out_channels, nspr_out_features)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

