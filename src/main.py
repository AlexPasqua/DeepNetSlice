from typing import Optional

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

from callbacks import AcceptanceRatioCallback
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
    create_HADRL_PSN_file(
        path=psn_path,
        n_CDCs=3,
        n_EDCs=10,
        n_servers_per_DC=(10, 5, 2)
    )

    base_tr_env = NetworkSimulator(
        psn_file=psn_path,
        nsprs_path='../NSPRs/',
        nsprs_per_episode=5,
        nsprs_max_duration=100,
    )
    # tr_env = make_vec_env(
    #     env_id=gym.wrappers.TimeLimit, n_envs=4,
    #     env_kwargs=dict(env=base_tr_env, max_episode_steps=30),
    #     wrapper_class=ResetWithRandLoad,
    #     wrapper_kwargs=dict(
    #         min_perc=0.1,
    #         max_perc=0.7,
    #         same_for_all=False,
    #     )
    # )

    tr_env = make_vec_env(
        env_id=make_env,
        n_envs=1,
        env_kwargs=dict(time_limit=False,
                        reset_with_rand_load=False,
                        hadrl_nsprs=True,
                        hadrl_nsprs_kwargs=dict(nsprs_per_ep=200, load=0.5)),
    )

    model = A2C(policy=HADRLPolicy, env=tr_env, verbose=2, device='auto',
                learning_rate=0.01,
                n_steps=5,  # ogni quanti step fare un update
                gamma=0.99,
                ent_coef=0.0,
                tensorboard_log="../tb_logs/",
                policy_kwargs=dict(
                    psn=base_tr_env.psn,
                    features_extractor_class=HADRLFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        psn=base_tr_env.psn,
                        activation_fn=nn.functional.relu
                    )
                ))

    print(model.policy)

    # base_eval_env = NetworkSimulator(
    #     psn_file=psn_path,
    #     nsprs_path='../NSPRs/',
    #     nsprs_per_episode=5,
    #     nsprs_max_duration=30
    # )
    # eval_env = make_vec_env(
    #     env_id=gym.wrappers.TimeLimit,
    #     n_envs=1,
    #     env_kwargs=dict(env=base_eval_env, max_episode_steps=30),
    #     wrapper_class=ResetWithRandLoad,
    #     wrapper_kwargs=dict(
    #         min_perc=0.1,
    #         max_perc=0.7,
    #         same_for_all=False,
    #     )
    # )
    eval_env = make_vec_env(
        env_id=make_env,
        n_envs=1,
        env_kwargs=dict(time_limit=False,
                        reset_with_rand_load=False,
                        hadrl_nsprs=True,
                        hadrl_nsprs_kwargs=dict(nsprs_per_ep=2)),
    )

    list_of_callbacks = [
        AcceptanceRatioCallback(name="Acceptance ratio", verbose=2),
        EvalCallback(eval_env=eval_env, n_eval_episodes=1, warn=True,
                     eval_freq=50, deterministic=True, verbose=2,
                     callback_after_eval=AcceptanceRatioCallback(
                         name="Eval acceptance ratio",
                         verbose=2
                     ))
    ]

    model.learn(total_timesteps=100000,
                log_interval=1,
                callback=list_of_callbacks)

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()
    # env.render()
