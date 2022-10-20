import random
from typing import Dict
import math

import gym
import networkx as nx
import numpy as np
import torch as th
from torch import nn


class P2CLoadBalanceHeuristic(nn.Module):
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
            **kwargs
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
        H = th.zeros_like(x)
        heu_selected_servers = self.HEU(obs, self.n_servers_to_sample)
        if th.all(heu_selected_servers == -1):
            return H  # it means no selected action by the heuristic
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
        n_envs = obs['bw_avails'].shape[0]
        indexes = th.empty(n_envs, n_servers_to_sample, dtype=th.int)
        req_cpu = obs['cur_vnf_cpu_req']
        req_ram = obs['cur_vnf_ram_req']
        load_balances = th.empty(n_envs, n_servers_to_sample)
        for e in range(n_envs):
            for s in range(n_servers_to_sample):
                # actions (indexes of the servers in the servers list)
                indexes[e, s] = self.action_space.sample()
                # servers ids
                node_id = self.servers_map_idx_id[indexes[e, s].item()]
                # actual servers (nodes in the graph)
                node = self.psn.nodes[node_id]
                # compute the load balance of each server when placing the VNF
                cpu_load_balance = (node['availCPU'] - req_cpu[e]) / node['CPUcap']
                ram_load_balance = (node['availRAM'] - req_ram[e]) / node['RAMcap']
                load_balances[e, s] = cpu_load_balance + ram_load_balance

        # return the best server for each environment (the indexes)
        winners = th.argmax(load_balances, dim=1, keepdim=True)
        return th.gather(indexes, 0, winners)


class HADRLHeuristic(nn.Module):
    def __init__(
            self,
            action_space: gym.spaces.Space,
            servers_map_idx_id: Dict[int, int],
            psn: nx.Graph,
            bw_req_per_vl: int = 2000,
            n_servers_to_sample: int = 2,
            eta: float = 0.,
            xi: float = 1.,
            beta: float = 1.,  # TODO: when not 1, could cause NaNs
            **kwargs
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
        self.bw_req_per_vl = bw_req_per_vl
        self.n_servers_to_sample = n_servers_to_sample
        self.eta, self.xi, self.beta = eta, xi, beta
        self.prev_selected_servers = None
        self.n_envs = None

    def forward(self, x: th.Tensor, obs: th.Tensor) -> th.Tensor:
        self.n_envs = x.shape[0]
        if self.prev_selected_servers is None or self.n_envs != self.prev_selected_servers.shape[0]:
            self.prev_selected_servers = -th.ones(self.n_envs, dtype=th.int)
        max_values, max_idxs = th.max(x, dim=1)
        H = th.zeros_like(x)
        heu_selected_servers = self.HEU(obs, self.n_servers_to_sample)
        if th.all(heu_selected_servers == -1):
            # it means no selected action by the heuristic
            return H
        for e in range(self.n_envs):
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
        indexes = th.empty(self.n_envs, n_servers_to_sample, dtype=th.int)
        path_lengths = th.zeros(self.n_envs, n_servers_to_sample)
        all_actions = list(range(self.action_space.n))
        for e in range(self.n_envs):
            # random permutation of the actions
            all_actions = np.random.permutation(all_actions)
            for s in range(n_servers_to_sample):
                # instead of selecting first all the feasible servers and then
                # sampling on them, we first create a list of all the actions
                # (i.e. servers) in random order, then we start going through
                # the list and pick the first action which is feasible.
                # This way we don't run through all the servers avery time
                for i in range(s, len(all_actions)):
                    a = all_actions[i]
                    if self.action_is_feasible(a, obs, e):
                        indexes[e, s] = a
                        break
                    # if no action is feasible, return no choice form the heuristic
                    # (i.e. tensor of -1's)
                    if i == len(all_actions) - 1:
                        return -th.ones(self.n_envs, 1)

                # server ID
                server_id = self.servers_map_idx_id[indexes[e, s].item()]

                if self.prev_selected_servers[e] == -1:
                    path_lengths[e, s] = -math.inf
                else:
                    # if the server was the one selected for the prev VNF, choose it
                    if self.prev_selected_servers[e] == server_id:
                        path_lengths[e, s] = -math.inf
                        # self.prev_selected_servers[e] = server_id
                    else:
                        # evaluate bandwidth consumption when placing the current VNF on this server
                        path = nx.shortest_path(G=self.psn,
                                                source=self.prev_selected_servers[e].item(),
                                                target=server_id,
                                                weight=self.compute_link_weight,
                                                method='dijkstra')
                        path_lengths[e, s] = len(path)

        # return the best server for each environment (the indexes)
        winners = th.argmin(path_lengths, dim=1, keepdim=True)
        selected_servers = th.gather(indexes, 1, winners)
        self.prev_selected_servers = selected_servers.squeeze(dim=1)
        return selected_servers

    @staticmethod
    def action_is_feasible(a: int, obs: th.Tensor, env_idx: int):
        """ Check if it's feasible to place the current VNF on a specific server

        1. if a server has enough CPU and RAM to host this VNF and the next one
        (all VNFs are assumed to have identical requirements, if this is not the
        case, then you can see this as "if a server has enough CPU and RAM to
        host double the requirements of this VNF", like a greedy safety margin),
        then it is eligible.

        2. if a server has enough CPU and RAM to host only this VNF, then if it
        has enough bandwidth in its outgoing links to host the connection with
        the neighboring VNFs, then it is eligible.

        3. if a server does not have enough CPU or RAM to host the current VNF,
        then it is NOT eligible.

        :param a: action, i.e. a server index
        :param obs: instance of an observation from the environment
        :param env_idx: index of the environment (in case of vectorized envs)
        :return: true if the action is feasible, false otherwise
        """
        req_cpu = obs['cur_vnf_cpu_req'][env_idx].item()
        req_ram = obs['cur_vnf_ram_req'][env_idx].item()
        req_bw = obs['cur_vnf_bw_req'][env_idx].item()
        avail_cpu = obs['cpu_avails'][env_idx][a].item()
        avail_ram = obs['ram_avails'][env_idx][a].item()
        avail_bw = obs['bw_avails'][env_idx][a]

        if (avail_cpu >= 2 * req_cpu and avail_ram >= 2 * req_ram) or \
                (avail_cpu >= req_cpu and avail_ram >= req_ram and avail_bw >= req_bw):
            return True

        return False

    def compute_link_weight(self, source: int, target: int, link: dict):
        return 1 if link['availBW'] >= self.bw_req_per_vl else math.inf
