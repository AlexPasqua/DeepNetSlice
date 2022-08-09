from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from environments.network_simulator import Simulator


def evaluate_agent(model, env):
    # evaluation prior to training
    n_episodes = 100
    steps_per_episode = 50
    tot_episode_reward = mean_episode_reward = 0
    for episode in range(n_episodes):
        obs = env.reset()
        for step in range(steps_per_episode):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            tot_episode_reward += reward
            if done:
                mean_episode_reward += tot_episode_reward
                tot_episode_reward = 0
                break
    mean_episode_reward /= n_episodes
    return mean_episode_reward


if __name__ == '__main__':
    env = Simulator(psn_file='../PSNs/triangle.graphml',
                    nsprs_path='../NSPRs/dummy_NSPR_1.graphml')

    check_env(env)  # check if the environment conforms to gym's API

    # if this is not done, A2C model will automatically wrap the env in a 'Monitor' and a 'DummyVecEnv' wrappers
    env = make_vec_env(lambda: env, n_envs=1)

    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        n_steps=100,
        verbose=1
    )

    # evaluation prior to training
    mean_ep_reward_prior_train = evaluate_agent(model, env)

    model.learn(total_timesteps=10000, log_interval=10)

    # evaluation after training
    mean_ep_reward_after_train = evaluate_agent(model, env)

    env.close()

    print(f"mean episode reward: {mean_ep_reward_prior_train}")
    print(f"mean episode reward: {mean_ep_reward_after_train}")


