import numpy as np
from tqdm import tqdm
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from callbacks import PSNLoadCallback
from utils import make_env
from wrappers.reset_with_load import ResetWithRealisticLoad

if __name__ == '__main__':
    # load model
    model = A2C.load(
        path="/root/NSPR-simulator/wandb/run-20230103_155854-3o0vtz6x/files/model.zip",
        env=None,
        device='cpu',
        print_system_info=True,
        force_reset=True,   # True as default -> avoids unexpected behavior
    )

    # re-create env
    env = make_vec_env(
        env_id=make_env,
        n_envs=1,
        env_kwargs=dict(
            psn_path="../PSNs/waxman_20_servers.graphml",
            base_env_kwargs=dict(accumulate_reward=True),
            time_limit=True,
            time_limit_kwargs=dict(max_episode_steps=1000),
            hadrl_nsprs=True,
            hadrl_nsprs_kwargs=dict(
                nsprs_per_ep=1,
                vnfs_per_nspr=5,
                always_one=True
            ),
            # hadrl_nsprs_kwargs=dict(
            #     nsprs_per_ep=None,
            #     load=0.5
            # )
            reset_load_class=ResetWithRealisticLoad,
            reset_load_kwargs = dict(cpu_load=0.5),
            placement_state=True,
            dynamic_connectivity=True,
            dynamic_connectivity_kwargs=dict(link_bw=10_000),
        ),
        seed=12,
    )

    # cpu_load_callback = PSNLoadCallback(env, freq=300, verbose=2)
    # cpu_load_callback.init_callback(model)

    # evaluate model
    obs = env.reset()
    accepted = seen = 0
    # accept_ratio_per_ep = []
    tot_nsprs = 10000
    pbar = tqdm(total=tot_nsprs)  # progerss bar
    while seen < tot_nsprs:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # cpu_load_callback.on_step()
        # acceptance ratio
        if rewards[0] != 0.0:
            seen += 1
            pbar.update(1)
            if rewards[0] > 0.0:
                accepted += 1
        if done:
            # if seen != 0.:
            #     cur_ep_accept_ratio = accepted / seen
            #     accept_ratio_per_ep.append(cur_ep_accept_ratio)
            #     print(f"Current episode's acceptance ratio: {cur_ep_accept_ratio}")
            # accepted = seen = 0
            obs = env.reset()

    # print(f"Acceptance ratio: {np.mean(accept_ratio_per_ep)}")
    print(f"Acceptance ratio: {accepted / seen}")
