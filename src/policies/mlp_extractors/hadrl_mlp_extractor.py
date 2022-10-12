from typing import Tuple, Dict

import gym
import networkx as nx
import torch as th
from torch import nn


class P2CHeuristic(nn.Module):
    """ Layer executing the P2C heuristic """

    def __init__(
            self,
            action_space: gym.spaces.Space,
            servers_map_idx_id: Dict[int, int],
            psn: nx.Graph,
            n_servers_to_sample: int = 2,
            eta: float = 0.,
            xi: float = 1.,
            beta: float = 1.,  # TODO: when not 1, could cause NaNs
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
        self.n_servers_to_sample = n_servers_to_sample
        self.eta, self.xi, self.beta = eta, xi, beta

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        n_envs = x.shape[0]
        max_values, max_idxs = th.max(x, dim=1)
        heu_selected_servers = self.HEU(obs, self.n_servers_to_sample)
        H = th.zeros_like(x)
        for e in range(n_envs):
            heu_action = heu_selected_servers[e, :].item()
            H[e, heu_action] = max_values[e] - x[e, heu_action] + self.eta
        out = x + self.xi * th.pow(H, self.beta)
        return out

    def HEU(self, obs: th.Tensor, n_servers_to_sample: int) -> th.Tensor:
        """ P2C heuristic to select the servers where to place the current VNFs.
        Selects one server for each environment (in case of vectorized envs).

        :param obs: Observation
        :param n_servers_to_sample: number of servers to sample
        :return: indexes of the selected servers
        """
        n_envs = obs['bw_availabilities'].shape[0]
        # s1_idxs = th.empty(1, n_envs, dtype=th.int)
        # s2_idxs = th.empty(1, n_envs, dtype=th.int)
        indexes = th.empty(n_envs, n_servers_to_sample, dtype=th.int)
        req_cpu = obs['cur_vnf_cpu_req']
        req_ram = obs['cur_vnf_ram_req']
        # load_balance_1 = th.empty(1, n_envs)
        # load_balance_2 = th.empty(1, n_envs)
        load_balances = th.empty(n_envs, n_servers_to_sample)
        for e in range(n_envs):
            for s in range(n_servers_to_sample):
                # actions (indexes of the servers in the servers list)
                indexes[e, s] = self.action_space.sample()
                # servers ids
                node_id = self.servers_map_idx_id[indexes[e, s].item()]
                # actual servers (nodes in the graph)
                node = self.psn.nodes[node_id]
                # node1 = self.psn.nodes[id1]
                # node2 = self.psn.nodes[id2]
                # compute the load balance of each server when placing the VNF
                cpu_load_balance = (node['availCPU'] - req_cpu[e]) / node['CPUcap']
                ram_load_balance = (node['availRAM'] - req_ram[e]) / node['RAMcap']
                load_balances[e, s] = cpu_load_balance + ram_load_balance
        # indexes = th.cat((s1_idxs, s2_idxs), dim=0)

        # return the best server for each environment (the indexes)
        winners = th.argmax(load_balances, dim=1, keepdim=True)
        return th.gather(indexes, 0, winners)


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
        self.heuristic = P2CHeuristic(
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
