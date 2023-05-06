import reader
from typing import Optional

import gym
from stable_baselines3.common.env_util import make_vec_env

from utils import make_env


class Trainer:
    def __init__(
        self,
        psn_path: str,
        n_tr_envs: int,
        load_perc: float,
        time_limit: bool,
        max_ep_steps: int,
        reset_load_class: Optional[gym.Wrapper] = None,
        placement_state: bool = True,
        accumulate_rew: bool = True,
        discount_acc_rew: bool = True,
        dynamic_connectivity: int = False,
        dynamic_connectivity_kwargs: dict = dict(link_bw=10_000),
        generate_nsprs: bool = True,
        nsprs_per_ep: int = 1,
        vnfs_per_nspr: int = 5,
        always_one: bool = True,
        seed: Optional[int] = None,
    ):
        # checks on argumetns
        assert n_tr_envs > 0
        assert 0. <= load_perc < 1., "Training load must be a percentage between 0 and 1"

        # read PSN file
        psn = reader.read_psn(psn_path)

        # create trainin environment
        tr_env = make_vec_env(
            env_id=make_env,
            n_envs=n_tr_envs,
            env_kwargs=dict(
                psn_path=psn_path,
                base_env_kwargs=dict(
                    accumulate_reward=accumulate_rew,
                    discount_acc_rew=discount_acc_rew,
                ),
                time_limit=time_limit,
                time_limit_kwargs=dict(max_episode_steps=max_ep_steps),
                generate_nsprs=generate_nsprs,
                nsprs_gen_kwargs=dict(
                    nsprs_per_ep=nsprs_per_ep,
                    vnfs_per_nspr=vnfs_per_nspr,
                    load=load_perc,
                    always_one=always_one
                ),
                reset_load_class=reset_load_class,
                reset_load_kwargs=dict(cpu_load=load_perc),
                placement_state=placement_state,
                dynamic_connectivity=dynamic_connectivity,
                dynamic_connectivity_kwargs=dynamic_connectivity_kwargs
            ),
            seed=seed,
        )
