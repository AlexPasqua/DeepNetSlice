from typing import Tuple, Dict

import gym
import networkx as nx
import numpy as np
import torch as th
from torch import nn


class P2CHeuristic(nn.Module):
    """ Layer executing the P2C heuristic """

    def __init__(
            self,
            action_space: gym.spaces.Space,
            servers_map_idx_id: Dict[int, int],
            psn: nx.Graph,
            eta: float = 0.,
            xi: float = 1.,
            beta: float = 0.5,
    ):
        """ Constructor

        :param action_space: Action space
        :param servers_map_idx_id: map (dict) between servers indexes (agent's actions) and their ids
        :param psn: the env's physical substrate network
        :param eta: hyperparameter of the P2C heuristic
        :param xi: hyperparameter of the P2C heuristic
        :param beta: hyperparameter of the P2C heuristic
        """
        super().__init__()
        self.action_space = action_space
        self.servers_map_idx_id = servers_map_idx_id
        self.psn = psn
        self.eta, self.xi, self.beta = eta, xi, beta

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        n_envs = x.shape[0]
        max_values, max_idxs = th.max(x, dim=1)
        heu_selected_servers = self.HEU(obs)
        H = th.zeros_like(x)
        for i in range(n_envs):
            heu_action = heu_selected_servers[:, i].item()
            H[i, heu_action] = max_values[i] - x[i, heu_action] + self.eta
        out = x + self.xi * th.pow(H, self.beta)
        return out

    def HEU(self, obs: th.Tensor) -> th.Tensor:
        """ P2C heuristic to select the servers where to place the current VNFs.
        Selects one server for each environment (in case of vectorized envs).

        :param obs: Observation
        :return: indexes of the selected servers
        """
        n_envs = obs['bw_availabilities'].shape[0]
        s1_idxs = th.empty(1, n_envs, dtype=th.int)
        s2_idxs = th.empty(1, n_envs, dtype=th.int)
        req_cpu = obs['cur_vnf_cpu_req']
        req_ram = obs['cur_vnf_ram_req']
        load_balance_1 = th.empty(1, n_envs)
        load_balance_2 = th.empty(1, n_envs)
        for i in range(n_envs):
            # actions (indexes of the servers in the servers list)
            s1_idxs[:, i] = self.action_space.sample()
            s2_idxs[:, i] = self.action_space.sample()
            # servers ids
            id1 = self.servers_map_idx_id[s1_idxs[:, i].item()]
            id2 = self.servers_map_idx_id[s2_idxs[:, i].item()]
            # actual servers (nodes in the graph)
            node1 = self.psn.nodes[id1]
            node2 = self.psn.nodes[id2]
            # compute the load balance of each server when placing the VNF
            load_balance_1[:, i] = (node1['availCPU'] - req_cpu[i]) / node1['CPUcap'] + \
                                   (node1['availRAM'] - req_ram[i]) / node1['RAMcap']
            load_balance_2[:, i] = (node2['availCPU'] - req_cpu[i]) / node2['CPUcap'] + \
                                   (node2['availRAM'] - req_ram[i]) / node2['RAMcap']
        indexes = th.cat((s1_idxs, s2_idxs), dim=0)

        # return the best server for each environment (the indexes)
        winners = th.argmax(th.cat((load_balance_1, load_balance_2)), dim=0, keepdim=True)
        return th.gather(indexes, 0, winners)


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
        self.heuristic = P2CHeuristic(action_space, servers_map_idx_id, psn)

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
        # x = self.heuristic(x, obs)
        x = th.softmax(x, dim=1)
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

    def forward(self, x: th.Tensor) -> th.Tensor:
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
