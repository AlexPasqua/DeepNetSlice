import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from callbacks import PSNLoadCallback
from utils import make_env

if __name__ == '__main__':
    # load model
    model = A2C.load(
        path="../wandb/run-20221124_154757-m8atvrw5/files/model.zip",
        env=None,
        device='cuda:0',
        print_system_info=True,
        force_reset=True,   # True as default -> avoids unexpected behavior
    )

    # re-create env
    env = make_vec_env(
        env_id=make_env,
        n_envs=1,
        env_kwargs=dict(
            psn_path="../PSNs/hadrl_psn_1-10_1-6_1-4.graphml",
            time_limit=True,
            time_limit_kwargs=dict(max_episode_steps=1000),
            reset_load_class=None,
            hadrl_nsprs=True,
            hadrl_nsprs_kwargs=dict(nsprs_per_ep=None,
                                    load=0.5)
        ),
    )

    cpu_load_callback = PSNLoadCallback(env, freq=300, verbose=2)
    cpu_load_callback.init_callback(model)

    # evaluate model
    obs = env.reset()
    accepted = seen = 0.0
    accept_ratio_per_ep = []
    for i in range(10_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        cpu_load_callback.on_step()
        # acceptance ratio
        if rewards[0] != 0.0:
            seen += 1
            if rewards[0] > 0.0:
                accepted += 1
        if done:
            if seen != 0.:
                cur_ep_accept_ratio = accepted / seen
                accept_ratio_per_ep.append(cur_ep_accept_ratio)
                print(f"Current episode's acceptance ratio: {cur_ep_accept_ratio}")
            accepted = seen = 0
            obs = env.reset()

    print(f"Acceptance ratio: {np.mean(accept_ratio_per_ep)}")
