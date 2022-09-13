import torch
from stable_baselines3 import A2C
from torch import nn

from environments.network_simulator import NetworkSimulator
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
    model = A2C('MultiInputPolicy', env, verbose=1,
                policy_kwargs=dict(
                    activation_fn=torch.nn.Tanh,
                    net_arch=[n_nodes, dict(pi=[n_nodes], vf=[1])],
                    features_extractor_class=HADRLFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        psn=env.psn,
                        activation=nn.functional.relu
                    )
                ),
                device='cpu')

    print(model.policy)

    model.learn(total_timesteps=25000, log_interval=100)
    exit()

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
        # env.render()
