from typing import List, Tuple

import gym
import networkx as nx
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class HADRLFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor network form HA-DRL paper:
    https://ieeexplore.ieee.org/document/9632824

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    def __init__(self, observation_space: gym.Space, psn: nx.Graph, features_dim: int):
        super().__init__(observation_space, features_dim=features_dim)
        self.psn = psn
        edges = th.tensor(np.array(self.psn.edges).reshape((len(self.psn.edges), 2)), dtype=th.long)
        double_edges = th.cat((edges, th.flip(edges, dims=(1,))))
        self.edge_index = double_edges.t().contiguous()

        in_channels = 4
        out_channels = 100
        self.graph_conv = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.final_linear = Linear(
            in_features=len(self.psn.nodes) * out_channels,
            out_features=features_dim
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        num_nodes = len(observations['cpu_availabilities'][0])
        len_rollout_buffer = len(observations['cpu_availabilities'])
        num_features = 4

        x = th.empty(size=(len_rollout_buffer, num_nodes, num_features), dtype=th.float)
        x[:, :, 0] = observations['cpu_availabilities']
        x[:, :, 1] = observations['ram_availabilities']
        x[:, :, 2] = observations['bw_availabilities']
        x[:, :, 3] = observations['placement_state']

        out = self.graph_conv(x, self.edge_index).flatten(start_dim=1) #.unsqueeze(dim=0)
        out = self.final_linear(out)
        return out

