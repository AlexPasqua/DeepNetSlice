import stable_baselines3 as sb3
import torch
from gym.wrappers import TimeLimit
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from torch import nn

from callbacks import AcceptanceRatioCallback
from environments.network_simulator import NetworkSimulator
from policies.hadrl_policy import HADRLPolicy
from policies.features_extractors import HADRLFeaturesExtractor

if __name__ == '__main__':
    env = NetworkSimulator(
        psn_file='../PSNs/servers_box_with_central_router.graphml',
        nsprs_path='../NSPRs/',
        nsprs_per_episode=5,
        nsprs_max_duration=100,
        reset_load_perc=0.5
    )

    vec_env = make_vec_env(
        env_id=TimeLimit,
        n_envs=5,
        env_kwargs=dict(
            env=NetworkSimulator(
                psn_file='../PSNs/servers_box_with_central_router.graphml',
                nsprs_path='../NSPRs/',
                nsprs_per_episode=5,
                nsprs_max_duration=100),
            max_episode_steps=3,
        )
    )

    n_nodes = len(env.psn.nodes)

    model = A2C(policy=HADRLPolicy, env=vec_env, verbose=2, device='auto',
                learning_rate=0.001,
                n_steps=5,  # ogni quanti step fare un update
                gamma=0.99,
                ent_coef=0.01,
                tensorboard_log="../tb_logs/",
                policy_kwargs=dict(
                    psn=env.psn,
                    features_extractor_class=HADRLFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        psn=env.psn,
                        activation_fn=nn.functional.relu
                    )
                ))

    print(model.policy)

    eval_env = NetworkSimulator(
        psn_file='../PSNs/servers_box_with_central_router.graphml',
        nsprs_path='../NSPRs/',
        nsprs_per_episode=8,
        nsprs_max_duration=100,
        reset_load_perc=0.5
    )
    eval_env = TimeLimit(eval_env, max_episode_steps=3)
    eval_env = sb3.common.env_util.Monitor(eval_env)
    eval_env = make_vec_env(lambda: eval_env, n_envs=5)

    model.learn(total_timesteps=100000,
                log_interval=100,
                callback=[
                    AcceptanceRatioCallback(name="Acceptance ratio", verbose=2),
                    EvalCallback(eval_env=eval_env, n_eval_episodes=4, warn=True,
                                 eval_freq=1000, deterministic=True, verbose=2,
                                 callback_after_eval=AcceptanceRatioCallback(
                                     name="Eval acceptance ratio",
                                     verbose=2
                                 ))
                ],)

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()
    # env.render()
