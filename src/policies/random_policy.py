import torch as th
from stable_baselines3.common.policies import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # useless in this case, needed for non-abstract class
        pass

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """ Get the action according to the policy for a given observation.

        (BasePolicy has a more complex 'predict' method, but it calls an abstract '_predict',
        which is being implemented now and performs the actual prediction)

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions,
            but this is a random policy, so this parameter is ignored.
        :return: Taken action according to the policy
        """
        return th.tensor([self.action_space.sample()])
