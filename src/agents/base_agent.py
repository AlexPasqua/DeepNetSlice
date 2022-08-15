from abc import ABC
from typing import Type, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class BaseAgent(BaseAlgorithm, ABC):
    """ Base agent class """

    def __init__(
            self,
            policy: Type[BasePolicy],
            env: Union[GymEnv, str, None],
            policy_base: Type[BasePolicy],
            learning_rate: Union[float, Schedule],
            support_multiple_envs: bool = False,
    ):
        super().__init__(policy, env, policy_base, learning_rate, support_multi_env=support_multiple_envs)
