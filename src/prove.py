import torch
from stable_baselines3 import A2C
from torch import nn

from environments.network_simulator import NetworkSimulator
from policies.hadrl_policy import HADRLPolicy
from policy_nets import HADRLFeaturesExtractor

if __name__ == '__main__':
    # env = make_vec_env("CartPole-v1", n_envs=4)

    # model = A2C("MlpPolicy", env, verbose=1)
    env = NetworkSimulator(psn_file='../PSNs/triangle.graphml',
                           nsprs_path='../NSPRs/',
                           max_nsprs_per_episode=2,
                           max_steps_per_episode=100)

    # env = PreventInfeasibleActions(env)

    # env = make_vec_env(lambda: env, n_envs=1)

    n_nodes = len(env.psn.nodes)
    # model = A2C('MultiInputPolicy', env, verbose=1,
    #             policy_kwargs=dict(
    #                 activation_fn=torch.nn.Tanh,
    #                 net_arch=[n_nodes, dict(pi=[n_nodes], vf=[1])],
    #                 features_extractor_class=HADRLFeaturesExtractor,
    #                 features_extractor_kwargs=dict(
    #                     psn=env.psn,
    #                     activation=nn.functional.relu
    #                 )
    #             ),
    #             device='cpu',
    #             )

    model = A2C(policy=HADRLPolicy, env=env, verbose=1, device='cpu',
                learning_rate=0.001,
                n_steps=1,  # per ora abbiamo pochi nodi (tipo 3), quindi facciamo un update dopo ogni step
                gamma=0.99,
                policy_kwargs=dict(
                    psn=env.psn,
                    features_extractor_class=HADRLFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        psn=env.psn,
                        activation=nn.functional.relu
                    )
                ))

    print(model.policy)

    model.learn(total_timesteps=100, log_interval=1)
    exit()

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()
        # env.render()
