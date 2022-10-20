from typing import Callable, Dict, List, Optional, Type, Union, Tuple

import gym
import networkx as nx
import torch as th
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from torch import nn

from .features_extractors import HADRLFeaturesExtractor
from .mlp_extractors.hadrl_mlp_extractor import HADRLActorCriticNet


class HADRLPolicy(MultiInputActorCriticPolicy):
    """ Policy network from the paper HA-DRL [1]

    [1] https://ieeexplore.ieee.org/document/9632824
    """
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            psn: nx.Graph,
            servers_map_idx_id: Dict[int, int],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            gcn_layers_dims: Tuple[int] = (60,),
            nspr_out_features: int = 4,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
            *args,
            **kwargs,
    ):
        """
        :param observation_space: Observation space of the agent
        :param action_space: Action space of the agent
        :param lr_schedule: Learning rate schedule
        :param psn: Physical Service Network
        :param servers_map_idx_id: Mapping between servers' indexes and their IDs
        :param net_arch: architecture of the policy and value networks after the feature extractor
        :param activation_fn: Activation function
        :param gcn_layers_dims: Dimensions of the GCN layers
        :param nspr_out_features: Number of output features of the NSPR state
        :param use_heuristic: Whether to use the heuristic or not
        :param heu_kwargs: Keyword arguments for the heuristic
        """
        self.psn = psn
        self.gcn_layers_dims = gcn_layers_dims  # saved in an attribute for logging purposes
        self.gcn_out_channels = gcn_layers_dims[-1]
        self.nspr_out_features = nspr_out_features
        self.servers_map_idx_id = servers_map_idx_id
        self.use_heuristic = use_heuristic
        self.heu_kwargs = heu_kwargs

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
            observation_space, psn, th.tanh, gcn_layers_dims,
            nspr_out_features
        )
        self.value_features_extractor = HADRLFeaturesExtractor(
            observation_space, psn, th.relu, gcn_layers_dims,
            nspr_out_features
        )
        delattr(self, "features_extractor")  # remove the shared features extractor

        # TODO: check what this step actually does
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HADRLActorCriticNet(
            self.action_space, self.psn, self.servers_map_idx_id,
            self.features_dim, self.gcn_out_channels, self.nspr_out_features,
            self.use_heuristic, self.heu_kwargs
        )

    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the feature extractor(s)
        """
        assert self.policy_features_extractor is not None and \
               self.value_features_extractor is not None
        preprocessed_obs = preprocess_obs(obs, self.observation_space,
                                          normalize_images=self.normalize_images)
        policy_features = self.policy_features_extractor(preprocessed_obs)
        value_features = self.value_features_extractor(preprocessed_obs)
        return policy_features, value_features

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> \
            Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features, obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> \
            Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features, obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :return: the action distribution.
        """
        policy_features, _ = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features, obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, value_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return self.value_net(latent_vf)
