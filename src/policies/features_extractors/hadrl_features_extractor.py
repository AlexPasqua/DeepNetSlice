import gym
import networkx as nx
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class HADRLFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor network form HA-DRL paper:
    https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(
            self,
            observation_space: gym.Space,
            psn: nx.Graph,
            activation_fn: nn.functional,
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4
    ):
        """ Constructor

        :param observation_space: the observation space of the agent using this feature extractor
        :param psn: the PSN graph of the environment which the agent acts upon
        :param activation_fn: activation function to be used
        :param gcn_out_channels: dimension of the features vector of each node after the GCN
        :param nspr_out_features: dimension of the features vector of the NSPR state
        """
        self.activation = activation_fn
        self.n_nodes = len(psn.nodes)
        features_dim = gcn_out_channels * self.n_nodes + nspr_out_features
        super().__init__(observation_space, features_dim=features_dim)
        self.n_features = 4  # same value both for PSN and NSPR states, since they are both 4-dimensional
        edges = th.tensor(np.array(psn.edges).reshape((len(psn.edges), 2)), dtype=th.long)
        double_edges = th.cat((edges, th.flip(edges, dims=(1,))))
        self.edge_index = double_edges.t().contiguous()

        self.graph_conv = GCNConv(in_channels=self.n_features,
                                  out_channels=gcn_out_channels)

        self.nspr_fc = Linear(in_features=self.n_features,
                              out_features=nspr_out_features)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        len_rollout_buffer = len(observations['cpu_availabilities'])

        # features extraction of the PSN state
        psn_state = th.empty(size=(len_rollout_buffer, self.n_nodes, self.n_features), dtype=th.float)
        psn_state[:, :, 0] = observations['cpu_availabilities']
        psn_state[:, :, 1] = observations['ram_availabilities']
        psn_state[:, :, 2] = observations['bw_availabilities']
        psn_state[:, :, 3] = observations['placement_state']
        gcn_out = self.graph_conv(psn_state, self.edge_index).flatten(start_dim=1)
        # gcn_out = self.activation_fn(gcn_out)

        # features extraction of the NSPR state
        nspr_state = th.empty(size=(len_rollout_buffer, 1, self.n_features), dtype=th.float)
        nspr_state[:, :, 0] = observations['cur_vnf_cpu_req']
        nspr_state[:, :, 1] = observations['cur_vnf_ram_req']
        nspr_state[:, :, 2] = observations['cur_vnf_bw_req']
        nspr_state[:, :, 3] = observations['vnfs_still_to_place']
        nspr_fc_out = self.nspr_fc(nspr_state.flatten(start_dim=1))
        # nspr_fc_out = self.activation_fn(nspr_fc_out)

        # concatenation of the two features vectors
        global_out = th.cat((gcn_out, nspr_fc_out), dim=1)
        global_out = self.activation(global_out)

        return global_out
