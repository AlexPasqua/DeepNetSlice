import gym
from gym.spaces import Discrete

import spaces
from spaces import DiscreteWithNegatives


class PreventInfeasibleActions(gym.ActionWrapper):
    """
    Wrap a Discrete action space to ensure a taken action meets the resources requirements.
    If no action is possible, return -1 (unsuccessful action).
    """

    def __init__(self, env: gym.Env):
        base_action_space = env.action_space
        assert isinstance(base_action_space, Discrete)
        super().__init__(env)
        self._action_space = src.spaces.DiscreteWithNegatives(n=base_action_space.n + 1, start=-1)

    def action(self, action):
        # NOTE: 'action' is already in the new space (self._action_space)

        # recall 'action' is the index in the list of servers corresponding to the
        # chosen server where to place the current VNF.

        # check that 'action' meets the resources requirements
        return action

    def reverse_action(self, action):
        raise NotImplementedError
