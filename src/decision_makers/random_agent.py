from typing import Union, Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from custom_policies.random_policy import RandomPolicy
from decision_makers.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random agent:
    when deciding a physical node where to place a VNF, it chooses one at random
    """

    def __init__(
            self,
            env: Union[GymEnv, str, None],
            limited: bool = False
    ):
        policy = RandomPolicy
        policy_base = RandomPolicy
        learning_rate = 0.  # any value will do, it's not used
        super().__init__(policy, env, policy_base, learning_rate, limited, support_multiple_envs=False)
        self._setup_model()

    def _setup_model(self) -> None:
        """ Create networks, buffer and optimizers """
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def learn(self, total_timesteps: int, callback: MaybeCallback = None,
              log_interval: int = 100, tb_log_name: str = "run", eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1, n_eval_episodes: int = 5, eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True) -> "BaseAlgorithm":
        # nothing to learn, it's a random agent
        pass
