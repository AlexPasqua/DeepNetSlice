from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

import reader
from callbacks import CPULoadCallback
from callbacks import HParamCallback
from callbacks import AcceptanceRatioByStepsCallback
from callbacks.acceptance_ratio_callbacks import AcceptanceRatioByNSPRsCallback
from environments.network_simulator import NetworkSimulator
from heuristic_layers import HADRLHeuristic, P2CLoadBalanceHeuristic
from policies.hadrl_policy import HADRLPolicy
from utils import make_env, create_HADRL_PSN_file

if __name__ == '__main__':
    psn_path = '../PSNs/hadrl_psn.graphml'

    # create_HADRL_PSN_file(
    #     path=psn_path,
    #     # n_CDCs=2,
    #     # n_EDCs=6,
    #     # n_servers_per_DC=(5, 3, 2),
    #     # n_EDCs_per_CDC=3
    # )

    psn = reader.read_psn(psn_path)

    # training environment
    n_tr_envs = 1
    tr_nsprs_per_ep = None
    tr_load = 0.1
    tr_time_limit = True
    tr_max_ep_steps = 1000
    tr_env = make_vec_env(
        env_id=make_env,
        n_envs=n_tr_envs,
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
    n_eval_envs = 1
    eval_nsprs_per_ep = None
    eval_load = 0.1
    eval_time_limit = True
    eval_max_ep_steps = 1000
    eval_env = make_vec_env(
        env_id=make_env,
        n_envs=n_eval_envs,
        env_kwargs=dict(
            psn_path=psn_path,
            time_limit=eval_time_limit,
            time_limit_kwargs=dict(max_episode_steps=eval_max_ep_steps),
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
    policy_kwargs = dict(psn=psn,
                         net_arch=[dict(pi=[256, 128], vf=[256, 128, 64])],
                         servers_map_idx_id=tr_env.get_attr('servers_map_idx_id', 0)[0],
                         gcn_layers_dims=(60, 60, 60, 40, 20),
                         use_heuristic=use_heuristic,
                         heu_kwargs=heu_kwargs,)

    model = A2C(policy=policy, env=tr_env, verbose=2, device='cuda:0',
                learning_rate=0.001,
                n_steps=10,  # ogni quanti step fare un update
                gamma=0.99,
                ent_coef=0.001,
                max_grad_norm=0.9,
                use_rms_prop=True,
                tensorboard_log="../tb_logs_fixed-nsprs-generation/",
                policy_kwargs=policy_kwargs)

    print(model.policy)

    # define some training hyperparams
    tot_tr_steps = 40_000_000

    # wandb stuff
    config = {
        "policy name": policy.name,
        "total tr timesteps": tot_tr_steps,
        "n tr envs": n_tr_envs,
        "n eval envs": n_eval_envs,
        "NSPRs per training ep": tr_nsprs_per_ep,
        "max steps per tr ep": tr_max_ep_steps if tr_time_limit else None,
        "PSN load (tr)": tr_load,
        "NSPRs per eval ep": eval_nsprs_per_ep,
        "max steps per eval ep": eval_max_ep_steps if eval_time_limit else None,
        "PSN load (eval)": eval_load,
        "GCNs layers dims": policy_kwargs['gcn_layers_dims'],
        "mpl_extractor arch": policy_kwargs["net_arch"],
        "use heuristic": use_heuristic,
        **heu_kwargs,
    }
    wandb_run = wandb.init(
        project="Fixed NSPR's generation",
        dir="../",
        # name="Simpler HADRL-style PSN - branch main",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    # training callbacks
    list_of_callbacks = [
        AcceptanceRatioByStepsCallback(env=tr_env, name="Acceptance ratio (by steps)",
                                       steps_per_tr_phase=500, verbose=2),

        AcceptanceRatioByNSPRsCallback(env=tr_env, name="Train acceptance ratio (by NSPRs)",
                                       nsprs_per_tr_phase=100, verbose=2),

        HParamCallback(tr_env.num_envs, eval_env.num_envs, tr_nsprs_per_ep,
                       tr_load,
                       tr_max_ep_steps=tr_max_ep_steps if tr_time_limit else None,
                       eval_nsprs_per_ep=eval_nsprs_per_ep,
                       eval_psn_load=eval_load,
                       eval_max_ep_steps=eval_max_ep_steps if eval_time_limit else None,
                       use_heuristic=use_heuristic, heu_kwargs=heu_kwargs, ),

        WandbCallback(model_save_path=f"../models/{wandb_run.id}",
                      verbose=2,
                      model_save_freq=10_000),

        EvalCallback(eval_env=eval_env, n_eval_episodes=1, warn=True,
                     eval_freq=5_000, deterministic=False, verbose=2,
                     callback_after_eval=AcceptanceRatioByNSPRsCallback(
                         env=eval_env,
                         name="Eval acceptance ratio (by NSPRs)",
                         nsprs_per_tr_phase=1,  # must be 1 for eval (default value)
                         verbose=2
                     )),

        # NOTE: currently it works only if all the servers have the same max CPU cap
        # (routers & switches, that have no CPU, are not a problem)
        CPULoadCallback(env=tr_env, freq=200),
    ]

    # model training
    model.learn(total_timesteps=tot_tr_steps,
                log_interval=4,
                # tb_log_name="A2C_Adam",
                callback=list_of_callbacks)

    wandb_run.finish()
