import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CPULoadCallback(BaseCallback):
    """
    Class for logging the load of the PSN.

    NOTE: currently it works correctly only if all the nodes have the same
    maximum CPU capacity (if some don't have CPU at all it's fine)

    :param env: environment
    :param freq: logging frequency (in number of steps)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env: gym.Env, freq: int, verbose: int = 0):
        super(CPULoadCallback, self).__init__(verbose)
        self.env = env
        self.freq = freq

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            loads = []
            observations = self.env.get_attr('obs_dict')
            for obs in observations:
                loads.append(1. - np.mean(obs['cpu_avails']))
            avg_load = np.mean(loads)
            self.logger.record("Average CPU load of training envs", avg_load)
        return True
