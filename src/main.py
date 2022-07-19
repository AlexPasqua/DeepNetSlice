import gym
from gym.utils.env_checker import check_env

from simulator import Simulator

if __name__ == '__main__':
    sim = Simulator(psn_file='../PSNs/triangle.graphml',
                    nsprs_path='../NSPRs/',
                    decision_maker_type='random')
    check_env(sim)  # check if the environment conforms to gym's API

    done = True
    sim_steps = 10000
    for step in range(sim_steps):
        if done:
            init_obs = sim.reset()

        # TODO: this takes random actions as a placeholder
        action = sim.action_space.sample()
        while sim.psn.nodes[action]['NodeType'] != "server":
            action = sim.action_space.sample()

        obs, reward, done, info = sim.step(action)

    sim.close()
