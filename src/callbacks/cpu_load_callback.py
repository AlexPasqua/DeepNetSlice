import warnings

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
        # TODO: it works only if all the servers have the same max CPU capacity
        if self.n_calls % self.freq == 0:
            loads = []
            observations = self.env.get_attr('obs_dict')
            for e, obs in enumerate(observations):
                servers_cpu_avails = []
                for server_id in self.env.get_attr('servers_map_idx_id')[e].values():
                    if self.env.get_attr('psn')[e].nodes[server_id]['NodeType'] == 'server':
                        idx = self.env.get_attr('map_id_idx')[e][server_id]
                        servers_cpu_avails.append(obs['cpu_avails'][idx])
                avg_cpu_avail = np.mean(servers_cpu_avails)
                loads.append(1. - avg_cpu_avail)
            avg_load = np.mean(loads)
            if self.verbose > 1:
                print(f"Average CPU load of training envs: {avg_load}")
            try:
                self.logger.record("Average CPU load of training envs", avg_load)
            except AttributeError:
                warnings.warn("No logger for CPU load callback, data not being logged")
        return True
