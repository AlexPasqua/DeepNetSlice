import gym
from gym.utils.env_checker import check_env

from simulator import Simulator

if __name__ == '__main__':
    env = Simulator(psn_file='../PSNs/triangle.graphml',
                    nsprs_path='../NSPRs/',
                    decision_maker_type='random')
    check_env(env)  # check if the environment conforms to gym's API

    # sim = Simulator(psn_file="../../PSNs/triangle.graphml",
    #                 nsprs_path="../../NSPRs/",
    #                 decision_maker_type="random")
    # sim.start()
