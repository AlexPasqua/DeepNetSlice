import warnings

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PSNLoadCallback(BaseCallback):
    """
    Class for logging the load of the PSN.

    :param env: environment
    :param freq: logging frequency (in number of steps)
    :param cpu: if True, track CPU load
    :param ram: if True, track RAM load
    :param bw: if True, track BW load
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
            self,
            env: gym.Env,
            freq: int,
            cpu: bool = True,
            ram: bool = True,
            bw: bool = True,
            verbose: int = 0
    ):
        super(PSNLoadCallback, self).__init__(verbose)
        self.env = env
        self.freq = freq
        self.cpu, self.ram, self.bw = cpu, ram, bw

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            cpu_loads, ram_loads, bw_loads = [], [], []
            observations = self.env.get_attr('obs_dict')
            for e, obs in enumerate(observations):
                # get the available CPU and RAM for each server
                serv_cpu_avails, serv_ram_avails = [], []
                for idx in self.env.get_attr('servers_map_idx_id')[e].keys():
                    serv_cpu_avails.append(obs['cpu_avails'][idx])
                    serv_ram_avails.append(obs['ram_avails'][idx])
                avail_cpu_perc = np.sum(serv_cpu_avails) * self.env.get_attr('max_cpu')[e] / self.env.get_attr('tot_cpu_cap')[e]
                avail_ram_perc = np.sum(serv_ram_avails) * self.env.get_attr('max_ram')[e] / self.env.get_attr('tot_ram_cap')[e]
                cpu_loads.append(1. - avail_cpu_perc)
                ram_loads.append(1. - avail_ram_perc)
                # get the available BW for each link
                link_bw_avails_perc = []
                for link in self.env.get_attr('psn')[e].edges.values():
                    link_bw_avails_perc.append(link['availBW'] / link['BWcap'])
                bw_loads.append(1. - np.mean(link_bw_avails_perc))
            try:
                if self.cpu:
                    avg_cpu_load = np.mean(cpu_loads)
                    self.logger.record("Average CPU load of training envs", avg_cpu_load)
                if self.ram:
                    avg_ram_load = np.mean(ram_loads)
                    self.logger.record("Average RAM load of training envs", avg_ram_load)
                if self.bw:
                    avg_bw_load = np.mean(bw_loads)
                    self.logger.record("Average BW load of training envs", avg_bw_load)
                if self.verbose > 0:
                    try:
                        print(f"Average CPU load of training envs: {avg_cpu_load}")
                        print(f"Average RAM load of training envs: {avg_ram_load}")
                        print(f"Average BW load of training envs: {avg_bw_load}")
                    except NameError:
                        # in case some variables are not defined. It means we're not tracking that load
                        pass
            except AttributeError:
                warnings.warn("No logger for resources load callback, data not being logged")

        return True
