from typing import Optional

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

import reader
from callbacks import AcceptanceRatioCallback, HParamCallback
from environments.network_simulator import NetworkSimulator
from policies.features_extractors import HADRLFeaturesExtractor
from policies.hadrl_policy import HADRLPolicy
from utils import create_HADRL_PSN_file
from wrappers import ResetWithRandLoad, NSPRsGeneratorHADRL


psn_path = '../PSNs/hadrl_psn.graphml'


def make_env(
        base_env_kwargs: Optional[dict] = None,
        time_limit: bool = False,
        time_limit_kwargs: Optional[dict] = None,
        reset_with_rand_load: bool = False,
        reset_with_rand_load_kwargs: Optional[dict] = None,
        hadrl_nsprs: bool = False,
        hadrl_nsprs_kwargs: Optional[dict] = None,
):
    base_env_kwargs = {} if base_env_kwargs is None else base_env_kwargs
    time_limit_kwargs = {} if time_limit_kwargs is None else time_limit_kwargs
    reset_with_rand_load_kwargs = {} if reset_with_rand_load_kwargs is None else reset_with_rand_load_kwargs

    env = NetworkSimulator(
        psn_file=psn_path,
        **base_env_kwargs
        # nsprs_path='../NSPRs/',
        # nsprs_per_episode=50,
        # nsprs_max_duration=30,
    )
    if time_limit:
        # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = gym.wrappers.TimeLimit(env, **time_limit_kwargs)
    if reset_with_rand_load:
        # env = ResetWithRandLoad(env, min_perc=0.1, max_perc=0.7, same_for_all=False)
        env = ResetWithRandLoad(env, **reset_with_rand_load_kwargs)
    if hadrl_nsprs:
        # env = NSPRsGeneratorHADRL(env, nsprs_per_ep=200)
        env = NSPRsGeneratorHADRL(env, **hadrl_nsprs_kwargs)
    return env


if __name__ == '__main__':
    # create_HADRL_PSN_file(
    #     path=psn_path,
    #     n_CDCs=2,
    #     n_EDCs=3,
    #     n_servers_per_DC=(5, 3, 2),
    #     n_EDCs_per_CDC=2
    # )

    psn = reader.read_psn(psn_path)

    tr_nsprs_per_ep = 100
    tr_load = 0.5
    tr_time_limit = False
    tr_max_ep_steps = 100

    tr_env = make_vec_env(
        env_id=make_env,
        n_envs=5,
        env_kwargs=dict(time_limit=tr_time_limit,
                        # time_limit_kwargs=dict(max_episode_steps=100),
                        reset_with_rand_load=False,
                        hadrl_nsprs=True,
                        hadrl_nsprs_kwargs=dict(nsprs_per_ep=tr_nsprs_per_ep,
                                                load=tr_load)),
    )

    use_heuristic = True
    heu_kwargs = {'n_servers_to_sample': 10, 'eta': 0., 'xi': 0.7, 'beta': 1.}

    model = A2C(policy=HADRLPolicy, env=tr_env, verbose=2, device='auto',
                learning_rate=0.05,
                n_steps=20,  # ogni quanti step fare un update
                gamma=0.8,
                ent_coef=0.01,
                # tensorboard_log="../tb_logs3/",
                policy_kwargs=dict(
                    psn=psn,
                    servers_map_idx_id=tr_env.get_attr('servers_map_idx_id', 0)[0],
                    gcn_layers_dims=(60, 30, 10),
                    use_heuristic=use_heuristic,
                    heu_kwargs=heu_kwargs,
                ))

    print(model.policy)

    eval_time_limit = False
    eval_nsprs_per_ep = 100
    eval_load = 0.5
    eval_max_ep_steps = 100

    eval_env = make_vec_env(
        env_id=make_env,
        n_envs=1,
        env_kwargs=dict(time_limit=eval_time_limit,
                        reset_with_rand_load=False,
                        hadrl_nsprs=True,
                        hadrl_nsprs_kwargs=dict(nsprs_per_ep=eval_nsprs_per_ep,
                                                load=eval_load)),
    )

    list_of_callbacks = [
        AcceptanceRatioCallback(name="Acceptance ratio", verbose=2),

        HParamCallback(tr_env.num_envs, eval_env.num_envs, tr_nsprs_per_ep, tr_load,
                       tr_max_ep_steps=tr_max_ep_steps if tr_time_limit else None,
                       eval_nsprs_per_ep=eval_nsprs_per_ep,
                       eval_psn_load=eval_load,
                       eval_max_ep_steps=eval_max_ep_steps if eval_time_limit else None,
                       use_heuristic=use_heuristic, heu_kwargs=heu_kwargs,),

        EvalCallback(eval_env=eval_env, n_eval_episodes=1, warn=True,
                     eval_freq=1000, deterministic=True, verbose=2,
                     callback_after_eval=AcceptanceRatioCallback(
                         name="Eval acceptance ratio",
                         verbose=2
                     ))
    ]

    model.learn(total_timesteps=1000000,
                log_interval=100,
                callback=list_of_callbacks)

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()
    # env.render()
