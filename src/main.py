import gym
from gym.utils.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from simulator import Simulator

if __name__ == '__main__':
    env = Simulator(psn_file='../PSNs/triangle.graphml',
                    nsprs_path='../NSPRs/',
                    decision_maker_type='random')
    check_env(env)  # check if the environment conforms to gym's API

    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        n_steps=10,
    )

    model.learn(total_timesteps=10000, log_interval=10)

    exit()

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        # env.render()

    # done = True
    # sim_steps = 1000
    # for step in range(sim_steps):
    #     if done:
    #         init_obs = env.reset()
    #
    #     # TODO: this takes random actions as a placeholder
    #     action = env.action_space.sample()
    #     while env.psn.nodes[action]['NodeType'] != "server":
    #         action = env.action_space.sample()
    #
    #     obs, reward, done, info = env.step(action)

    env.close()
