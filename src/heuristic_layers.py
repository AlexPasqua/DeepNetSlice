import random
from typing import Dict

import gym
import networkx as nx
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
            # it means no selected action by the heuristic
            return H
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

    # @staticmethod
    # def _get_feasible_servers(obs: th.Tensor, env_idx: int) -> list:
    #     """ Get the list of eligible servers for the current VNF.
    #
    #     1. if a server has enough CPU and RAM to host this VNF and the next one
    #     (all VNFs are assumed to have identical requirements, if this is not the
    #     case, then you can see this as "if a server has enough CPU and RAM to
    #     host double the requirements of this VNF", like a greedy safety margin),
    #     then it is eligible.
    #
    #     2. if a server has enough CPU and RAM to host only this VNF, then if it
    #     has enough bandwidth in its outgoing links to host the connection with
    #     the neighboring VNFs, then it is eligible.
    #
    #     3. if a server does not have enough CPU or RAM to host the current VNF,
    #     then it is NOT eligible.
    #
    #     :param obs: instance of an observation from the environment
    #     :param env_idx: index of the environment (in case of vectorized envs)
    #     :return: the list of eligible servers to be sampled by the heuristic
    #     """
    #     req_cpu = obs['cur_vnf_cpu_req'][env_idx].item()
    #     req_ram = obs['cur_vnf_ram_req'][env_idx].item()
    #     req_bw = obs['cur_vnf_bw_req'][env_idx].item()
    #
    #     # iterate over servers and save the eligible ones
    #     eligible_ones = []
    #     for s in range(len(obs['cpu_avails'][env_idx])):
    #         avail_cpu = obs['cpu_avails'][env_idx][s].item()
    #         avail_ram = obs['ram_avails'][env_idx][s].item()
    #         avail_bw = obs['bw_avails'][env_idx][s].item()
    #
    #         # if the node is a server (and not a router or switch)
    #         if avail_cpu > 0 and avail_ram > 0:
    #             # check if the server has enough CPU and RAM to host the current VNF
    #             # and the next one (or double the requirements of the current VNF)
    #             if avail_cpu >= 2 * req_cpu and avail_ram >= 2 * req_ram:
    #                 eligible_ones.append(s)
    #
    #             # check if the server has enough CPU and RAM to host only the current
    #             # VNF, but has enough bandwidth in its outgoing links to host the
    #             # connection with the neighboring VNFs
    #             elif avail_cpu >= req_cpu and avail_ram >= req_ram and avail_bw >= req_bw:
    #                 eligible_ones.append(s)
    #
    #     return eligible_ones
