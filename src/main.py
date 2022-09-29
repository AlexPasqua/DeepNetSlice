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
from wrappers import ResetWithRandLoad, HadrlDataGenerator


psn_path = '../PSNs/hadrl_psn.graphml'


def make_env():
    env = NetworkSimulator(
        psn_file=psn_path,
        nsprs_path='../NSPRs/',
        nsprs_per_episode=5,
        nsprs_max_duration=100,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=30)
    env = ResetWithRandLoad(env, min_perc=0.1, max_perc=0.7, same_for_all=False)
    # env = HadrlDataGenerator(env, path='../PSNs/hadrl_psn.graphml')
    return env


if __name__ == '__main__':
    create_HADRL_PSN_file(path=psn_path)

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

    tr_env = make_vec_env(env_id=make_env, n_envs=4)

    n_nodes = len(base_tr_env.psn.nodes)

    model = A2C(policy=HADRLPolicy, env=tr_env, verbose=2, device='auto',
                learning_rate=0.001,
                n_steps=5,  # ogni quanti step fare un update
                gamma=0.99,
                ent_coef=0.01,
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

    base_eval_env = NetworkSimulator(
        psn_file=psn_path,
        nsprs_path='../NSPRs/',
        nsprs_per_episode=5,
        nsprs_max_duration=30
    )
    eval_env = make_vec_env(
        env_id=gym.wrappers.TimeLimit,
        n_envs=4,
        env_kwargs=dict(env=base_eval_env, max_episode_steps=30),
        wrapper_class=ResetWithRandLoad,
        wrapper_kwargs=dict(
            min_perc=0.1,
            max_perc=0.7,
            same_for_all=False,
        )
    )

    list_of_callbacks = [
        AcceptanceRatioCallback(name="Acceptance ratio", verbose=2),
        # EvalCallback(eval_env=eval_env, n_eval_episodes=4, warn=True,
        #              eval_freq=500, deterministic=True, verbose=2,
        #              callback_after_eval=AcceptanceRatioCallback(
        #                  name="Eval acceptance ratio",
        #                  verbose=2
        #              ))
    ]

    model.learn(total_timesteps=100000,
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
