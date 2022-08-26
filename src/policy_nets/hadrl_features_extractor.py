from typing import List, Tuple

import gym
import networkx as nx
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.nn import GCNConv


class HADRLFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor network form HA-DRL paper:
    https://ieeexplore.ieee.org/document/9632824

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    def __init__(self, observation_space: gym.Space, edges: List[Tuple[int, int]], features_dim: int):
        super().__init__(observation_space, features_dim=features_dim)
        edges = th.tensor(np.array(edges).reshape((len(edges), 2)), dtype=th.long)
        double_edges = th.cat((edges, th.flip(edges, dims=(1,))))
        self.edge_index = double_edges.t().contiguous()

        self.graph_conv = GCNConv(
            in_channels=4,
            out_channels=features_dim,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        num_nodes = len(observations['cpu_availabilities'][0])
        num_features = 4

        x = th.empty(size=(num_nodes, num_features), dtype=th.float)
        x[:, 0] = observations['cpu_availabilities']
        x[:, 1] = observations['ram_availabilities']
        x[:, 2] = observations['bw_availabilities']
        x[:, 3] = observations['placement_state']

        out = self.graph_conv(x, self.edge_index).flatten().unsqueeze(dim=0)
        return out

