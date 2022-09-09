import gym
import networkx as nx
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class HADRLFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor network form HA-DRL paper:
    https://ieeexplore.ieee.org/document/9632824
    """

    def __init__(self, observation_space: gym.Space, psn: nx.Graph,
                 features_dim: int, gcn_out_channels: int = 60):
        """ Constructor

        :param observation_space: the observation space of the agent using this feature extractor
        :param psn: the PSN graph of the environment which the agent acts upon
        :param features_dim: the dimension of the extracted features
        :param gcn_out_channels: the dimension of the features vector of each node after the GCN
        """
        super().__init__(observation_space, features_dim=features_dim)
        self.n_features = 4
        self.n_nodes = len(psn.nodes)
        edges = th.tensor(np.array(psn.edges).reshape((len(psn.edges), 2)), dtype=th.long)
        double_edges = th.cat((edges, th.flip(edges, dims=(1,))))
        self.edge_index = double_edges.t().contiguous()

        gcn_out_channels = 100
        self.graph_conv = GCNConv(
            in_channels=self.n_features,
            out_channels=gcn_out_channels,
        )

        self.gcn_fc = Linear(
            in_features=len(psn.nodes) * gcn_out_channels,
            out_features=features_dim
        )

        self.nspr_fc = Linear(in_features=self.n_features, out_features=4)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        len_rollout_buffer = len(observations['cpu_availabilities'])

        psn_state = th.empty(size=(len_rollout_buffer, self.n_nodes, self.n_features), dtype=th.float)
        psn_state[:, :, 0] = observations['cpu_availabilities']
        psn_state[:, :, 1] = observations['ram_availabilities']
        psn_state[:, :, 2] = observations['bw_availabilities']
        psn_state[:, :, 3] = observations['placement_state']
        gcn_out = self.graph_conv(psn_state, self.edge_index).flatten(start_dim=1)  #.unsqueeze(dim=0)
        gcn_out = self.gcn_fc(gcn_out)

        nspr_state = th.tensor([observations['cur_vnf_cpu_req'],
                                observations['cur_vnf_ram_req'],
                                observations['cur_vnf_bw_req'],
                                observations['vnfs_still_to_place']])
        nspr_fc_out = self.nspr_fc(nspr_state)

        return gcn_out

