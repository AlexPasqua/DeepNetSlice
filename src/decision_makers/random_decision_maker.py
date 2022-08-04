import random
from typing import Tuple, Union, Type, Optional

import networkx as nx
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback

from decision_makers.base_decision_maker import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random agent:
    when deciding a physical node where to place a VNF, it chooses one at random
    """

    def __init__(
            self,
            policy: Type[BasePolicy],
            env: Union[GymEnv, str, None],
            policy_base: Type[BasePolicy],
            learning_rate: Union[float, Schedule],
            limited: bool = False
    ):
        super().__init__(policy, env, policy_base, learning_rate, limited)

    def _setup_model(self) -> None:
        """ Normally used to create networks, buffer and optimizers.
            Here, nothing to do (but method needed for non-abstract class)
        """
        pass

    def learn(self, total_timesteps: int, callback: MaybeCallback = None,
              log_interval: int = 100, tb_log_name: str = "run", eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1, n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        # nothing to learn, it's a random agent
        pass

    def decide_next_node(self, psn: nx.Graph, vnf: dict) -> Tuple[int, dict]:
        """
        Given a PSN a NSPR, decides where to place the first VNF that hasn't been placed onto the PSN yet.
        For RandomAgent, it simply chooses at random a physical node that satisfies the resources requirements

        :param psn: a graph representation of the PSN
        :param vnf: a VNF to be placed on the PSN
        :return: the ID and the physical node itself onto which to place the VNF (if not possible, returns -1, {})
        """
        psn_servers_ids = [node_id for node_id, node in psn.nodes.items() if node['NodeType'] == "server"]
        selected_id = psn_servers_ids.pop(random.randint(0, len(psn_servers_ids) - 1))
        while not self.resources_reqs_satisfied(psn.nodes[selected_id], vnf) and len(psn_servers_ids) > 0:
            selected_id = psn_servers_ids.pop(random.randint(0, len(psn_servers_ids) - 1))
            if len(psn_servers_ids) == 0:
                # no physical nodes satisfy the resources requirements --> fail to accept NSPR
                return -1, {}
        return selected_id, psn.nodes[selected_id]
