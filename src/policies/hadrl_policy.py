from typing import Callable, Dict, List, Optional, Type, Union

import gym
import networkx as nx
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from torch import nn

from .features_extractors import HADRLFeaturesExtractor
from .mlp_extractors.hadrl_mlp_extractor import HADRLActorCriticNet


class HADRLPolicy(MultiInputActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            psn: nx.Graph,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            gcn_out_channels: int = 60,
            nspr_out_features: int = 4,
            *args,
            **kwargs,
    ):

        self.psn = psn
        self.gcn_out_channels = gcn_out_channels
        self.nspr_out_features = nspr_out_features

        super(HADRLPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # non-shared features extractors for the actor and the critic
        self.policy_features_extractor = HADRLFeaturesExtractor(
            observation_space, psn, activation_fn, gcn_out_channels,
            nspr_out_features
        )
        self.value_features_extractor = HADRLFeaturesExtractor(
            observation_space, psn, activation_fn, gcn_out_channels,
            nspr_out_features
        )

        # TODO: check what this step actually does
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HADRLActorCriticNet(
            self.observation_space, self.psn, self.features_dim,
            self.gcn_out_channels, self.nspr_out_features
        )
