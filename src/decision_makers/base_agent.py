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
            limited: bool = False,
            support_multiple_envs: bool = False,
    ):
        super().__init__(policy, env, policy_base, learning_rate, support_multi_env=support_multiple_envs)
        self.limited = limited  # if true, the agent will only choose physical nodes/links with enough available resources

    @staticmethod
    def resources_reqs_satisfied(physical_node: dict, vnf: dict):
        """ Checks whether the resources requirements of a certain VNF are satisfied by a certain physical node on the PSN

        :param physical_node: physical node on the PSN
        :param vnf: virtual network function requiring some resources
        :return: True if the resources required by the VNF are available on the physical node, else False
        """
        if physical_node['availCPU'] >= vnf['reqCPU'] and physical_node['availRAM'] >= vnf['reqRAM']:
            return True
        return False
