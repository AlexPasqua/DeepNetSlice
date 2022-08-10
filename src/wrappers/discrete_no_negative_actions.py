import gym
from gym.spaces import Discrete

from spaces import EnhancedDiscrete


class DiscreteNoNegativeActions(gym.ActionWrapper):
    """ Wrap a EnhancedDiscrete action space to ensure alla actions are non-negative """

    def __init__(self, env: gym.Env):
        base_action_space = env.action_space
        assert isinstance(base_action_space, EnhancedDiscrete)
        super().__init__(env)
        n_actions_to_remove = -base_action_space.start if base_action_space.start < 0 else 0
        self._action_space = Discrete(base_action_space.n - n_actions_to_remove)

    def action(self, action):
        return action   # it is already in the new space (self._action_space)

    def reverse_action(self, action):
        raise NotImplementedError
