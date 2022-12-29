import gym
import numpy as np

from gym.spaces import Dict, Box


class RemovePlacementState(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        ONE_BILLION = 1_000_000_000  # constant for readability
        n_nodes = len(self.unwrapped.psn.nodes)
        self.observation_space = Dict({
            # PSN STATE
            'cpu_avails': Box(low=0., high=1., shape=(n_nodes,), dtype=np.float32),
            'ram_avails': Box(low=0., high=1., shape=(n_nodes,), dtype=np.float32),
            # for each physical node, sum of the BW of the physical links connected to it
            'bw_avails': Box(low=0., high=1., shape=(n_nodes,), dtype=np.float32),

            # NSPR STATE
            # note: apparently it's not possible to pass "math.inf" or "sys.maxsize" as a gym.spaces.Box's high value
            'cur_vnf_cpu_req': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=np.float32),
            'cur_vnf_ram_req': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=np.float32),
            # sum of the required BW of each VL connected to the current VNF
            'cur_vnf_bw_req': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=np.float32),
            'vnfs_still_to_place': Box(low=0, high=ONE_BILLION, shape=(1,), dtype=int),
        })

    def observation(self, obs):
        """returns the observation without the placement state """
        new_obs = {
            'cpu_avails': obs['cpu_avails'],
            'ram_avails': obs['ram_avails'],
            'bw_avails': obs['bw_avails'],
            'cur_vnf_cpu_req': obs['cur_vnf_cpu_req'],
            'cur_vnf_ram_req': obs['cur_vnf_ram_req'],
            'cur_vnf_bw_req': obs['cur_vnf_bw_req'],
            'vnfs_still_to_place': obs['vnfs_still_to_place'],
        }
        return new_obs