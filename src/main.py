from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

import reader
from callbacks import AcceptanceRatioCallback, HParamCallback
from heuristic_layers import HADRLHeuristic, P2CLoadBalanceHeuristic
from policies.hadrl_policy import HADRLPolicy
from utils import make_env, create_HADRL_PSN_file

if __name__ == '__main__':
    psn_path = '../PSNs/simple_hadrl_psn.graphml'

    # create_HADRL_PSN_file(
    #     path=psn_path,
    #     n_CDCs=2,
    #     n_EDCs=6,
    #     n_servers_per_DC=(5, 3, 2),
    #     n_EDCs_per_CDC=3
    # )

    psn = reader.read_psn(psn_path)

    # training environment
    tr_nsprs_per_ep = 8
    tr_load = 0.5
    tr_time_limit = False
    tr_max_ep_steps = 100
    tr_env = make_vec_env(
        env_id=make_env,
        n_envs=10,
        env_kwargs=dict(
            psn_path=psn_path,
            time_limit=tr_time_limit,
            time_limit_kwargs=dict(max_episode_steps=tr_max_ep_steps),
            reset_with_rand_load=False,
            hadrl_nsprs=True,
            hadrl_nsprs_kwargs=dict(nsprs_per_ep=tr_nsprs_per_ep,
                                    load=tr_load)
        ),
    )

    # evaluation environment
    eval_time_limit = False
    eval_nsprs_per_ep = 8
    eval_load = 0.5
    eval_max_ep_steps = 100
    eval_env = make_vec_env(
        env_id=make_env,
        n_envs=1,
        env_kwargs=dict(
            psn_path=psn_path,
            time_limit=eval_time_limit,
            reset_with_rand_load=False,
            hadrl_nsprs=True,
            hadrl_nsprs_kwargs=dict(nsprs_per_ep=eval_nsprs_per_ep,
                                    load=eval_load)
        ),
    )

    # model definition
    use_heuristic = False
    heu_kwargs = {'n_servers_to_sample': 10, 'heu_class': P2CLoadBalanceHeuristic,
                  'eta': 0.05, 'xi': 1., 'beta': 1.}
    policy = HADRLPolicy

    model = A2C(policy=policy, env=tr_env, verbose=2, device='cuda:0',
                learning_rate=0.05,
                n_steps=10,  # ogni quanti step fare un update
                gamma=0.8,
                ent_coef=0.001,
                use_rms_prop=True,
                # tensorboard_log="../tb_logs_new_eval_rew_test/",
                policy_kwargs=dict(
                    psn=psn,
                    servers_map_idx_id=tr_env.get_attr('servers_map_idx_id', 0)[0],
                    gcn_layers_dims=(60, 60, 60, 40, 20),
                    use_heuristic=use_heuristic,
                    heu_kwargs=heu_kwargs,
                ))

    print(model.policy)

    # define some training hyperparams
    tot_tr_steps = 4_000_000

    # wandb stuff
    config = {
        "policy_type": policy,
        "total_timesteps": tot_tr_steps,
    }
    # wandb_run = wandb.init(
    #     project="New eval rew test",
    #     dir="../",
    #     name="Simpler HADRL-style PSN - branch main",
    #     config=config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     # monitor_gym=True,  # auto-upload the videos of agents playing the game
    #     # save_code=True,  # optional
    # )

    # training callbacks
    list_of_callbacks = [
        AcceptanceRatioCallback(name="Acceptance ratio", verbose=2),

        HParamCallback(tr_env.num_envs, eval_env.num_envs, tr_nsprs_per_ep,
                       tr_load,
                       tr_max_ep_steps=tr_max_ep_steps if tr_time_limit else None,
                       eval_nsprs_per_ep=eval_nsprs_per_ep,
                       eval_psn_load=eval_load,
                       eval_max_ep_steps=eval_max_ep_steps if eval_time_limit else None,
                       use_heuristic=use_heuristic, heu_kwargs=heu_kwargs, ),

        # WandbCallback(model_save_path=f"../models_prova/{wandb_run.id}", verbose=2),

        EvalCallback(eval_env=eval_env, n_eval_episodes=2, warn=True,
                     eval_freq=100, deterministic=True, verbose=2,
                     callback_after_eval=AcceptanceRatioCallback(
                         name="Eval acceptance ratio",
                         verbose=2
                     ))
    ]

    # model training
    model.learn(total_timesteps=tot_tr_steps,
                log_interval=10,
                # tb_log_name="A2C_Adam",
                callback=list_of_callbacks)

    # wandb_run.finish()

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         env.reset()
    # env.render()
