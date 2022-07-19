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

        obs, reward, done, info = sim.step(action)

    begin_new_episode = True
    nsprs_waiting_list = []
    sim_steps = 100
    for step in range(sim_steps):
        # add eventual newly arrived NSPRs to the list of NSPRs to be evaluated and skip if there are none
        nsprs_waiting_list += sim.nsprs.get(step, [])
        if len(nsprs_waiting_list) == 0:
            continue

        # pop a NSPR from the list of NSPRs that arrived already
        cur_nspr = nsprs_waiting_list.pop(0)
        sim.cur_nspr = cur_nspr
        if begin_new_episode:
            obs = sim.reset()   # reset the environment
            begin_new_episode = False

        # TODO: this takes random actions as a placeholder
        action = sim.action_space.sample()

        sim.step(action=action)

    sim.close()
