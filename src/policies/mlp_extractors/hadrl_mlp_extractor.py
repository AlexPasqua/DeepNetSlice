from typing import Tuple, Dict

import gym
import networkx as nx
import numpy as np
import torch as th
from torch import nn


class P2CHeuristic(nn.Module):
    def __init__(
            self,
            action_space: gym.spaces.Space,
            servers_map_idx_id: Dict[int, int],
    ):
        super().__init__()
        self.action_space = action_space
        self.servers_map_idx_id = servers_map_idx_id

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        max_values, max_idxs = th.max(x, dim=1)
        pass

    def HEU(self, obs: th.Tensor):
        # actions (indexes of the servers in the servers list)
        s1_idx = self.action_space.sample()
        s2_idx = self.action_space.sample()

        # servers ids
        s1_id = self.servers_map_idx_id[s1_idx]
        s2_id = self.servers_map_idx_id[s2_idx]

        # actual servers (nodes in the graph)
        node1 = self.psn.nodes[s1_id]
        node2 = self.psn.nodes[s2_id]

        # compute the load balance of each server when placing the VNF
        # TODO: check dimension of obs with vectorized env
        req_cpu = obs['cur_vnf_cpu_req']
        req_ram = obs['cur_vnf_ram_req']
        load_balance_1 = (node1['availCPU'] - req_cpu) / node1['CPUcap'] + \
                         (node1['availRAM'] - req_ram) / node1['RAMcap']
        load_balance_2 = (node2['availCPU'] - req_cpu) / node2['CPUcap'] + \
                         (node2['availRAM'] - req_ram) / node2['RAMcap']

        # return the best server
        indexes = [s1_idx, s2_idx]
        winner = np.argmax([load_balance_1, load_balance_2])
        return indexes[winner]


class HADRLActor(nn.Module):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            psn: nx.Graph,
            servers_map_idx_id: Dict[int, int],
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4):
        super().__init__()
        # self.features_extractor = HADRLFeaturesExtractor(observation_space, psn, nn.Tanh, gcn_out_channels, nspr_out_features)

        n_nodes = len(psn.nodes)
        self.linear = nn.Linear(
                in_features=n_nodes * gcn_out_channels + nspr_out_features,
                out_features=n_nodes)
        self.heuristic = P2CHeuristic(action_space, servers_map_idx_id)

        # self.final_fcs = nn.Sequential(
        #     nn.Linear(
        #         in_features=n_nodes * gcn_out_channels + nspr_out_features,
        #         out_features=n_nodes),
        #     nn.Tanh(),
        #     # P2C heuristic layer
        #     P2CHeuristic(action_space, servers_map_idx_id),
        #     # nn.Softmax(dim=1)
        # )

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        x = th.tanh(self.linear(x))
        x = self.heuristic(x, obs)
        return x


class HSDRLCritic(nn.Module):
    def __init__(self, observation_space: gym.Space, psn: nx.Graph,
                 gcn_out_channels: int = 60,
                 nspr_out_features: int = 4):
        super().__init__()
        # self.features_extractor = HADRLFeaturesExtractor(observation_space, psn, nn.ReLU, gcn_out_channels, nspr_out_features)

        n_nodes = len(psn.nodes)
        self.final_fcs = nn.Sequential(
            nn.Linear(
                in_features=n_nodes * gcn_out_channels + nspr_out_features,
                out_features=n_nodes),
            nn.ReLU(),
            nn.Linear(in_features=n_nodes, out_features=1),
            nn.ReLU()
        )

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        # x = self.features_extractor(x)
        return self.final_fcs(x)


class HADRLActorCriticNet(nn.Module):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            psn: nx.Graph,
            servers_map_idx_id: Dict[int, int],
            feature_dim: int,
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4
    ):
        super(HADRLActorCriticNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = len(psn.nodes)
        self.latent_dim_vf = 1

        # policy network
        self.policy_net = HADRLActor(observation_space, action_space, psn,
                                     servers_map_idx_id,
                                     gcn_out_channels, nspr_out_features)

        # value network
        self.value_net = HSDRLCritic(observation_space, psn, gcn_out_channels,
                                     nspr_out_features)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor, obs: th.Tensor) -> th.Tensor:
        return self.policy_net(features, obs)

    def forward_critic(self, features: th.Tensor, obs: th.Tensor) -> th.Tensor:
        return self.value_net(features, obs)
