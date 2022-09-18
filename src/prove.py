import torch
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from torch import nn

from callbacks import AcceptanceRatioCallback
from environments.network_simulator import NetworkSimulator
from policies.hadrl_policy import HADRLPolicy
from policies.features_extractors import HADRLFeaturesExtractor

if __name__ == '__main__':
    # env = make_vec_env("CartPole-v1", n_envs=4)

    # model = A2C("MlpPolicy", env, verbose=1)
    env = NetworkSimulator(
        psn_file='../PSNs/servers_box_with_central_router.graphml',
        nsprs_path='../NSPRs/',
        nsprs_per_episode=5,
        max_steps_per_episode=100, )

    # env = make_vec_env(lambda: env, n_envs=1)

    n_nodes = len(env.psn.nodes)
    # model = A2C('MultiInputPolicy', env, verbose=1,
    #             policy_kwargs=dict(
    #                 activation_fn=torch.nn.Tanh,
    #                 net_arch=[n_nodes, dict(pi=[n_nodes], vf=[1])],
    #                 features_extractor_class=HADRLFeaturesExtractor,
    #                 features_extractor_kwargs=dict(
    #                     psn=env.psn,
    #                     activation_fn=nn.functional.relu
    #                 )
    #             ),
    #             device='cpu',
    #             )

    model = A2C(policy=HADRLPolicy, env=env, verbose=2, device='auto',
                learning_rate=0.001,
                n_steps=5,  # ogni quanti step fare un update
                gamma=0.99,
                ent_coef=0.,
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
        nsprs_per_episode=5,
        max_steps_per_episode=100, )

    model.learn(total_timesteps=100000, log_interval=100,
                callback=AcceptanceRatioCallback(verbose=1),
                eval_env=eval_env,
                eval_freq=1000,
                n_eval_episodes=4, )
    exit()

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()
    # env.render()
