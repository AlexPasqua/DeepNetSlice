import networkx as nx
import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from wandb.integration.sb3 import WandbCallback

import reader
from callbacks import PSNLoadCallback, HParamCallback, AcceptanceRatioByNSPRsCallback, SeenNSPRsCallback
from heuristic_layers import P2CLoadBalanceHeuristic
from policies.features_extractors import HADRLFeaturesExtractor
from policies.hadrl_policy import HADRLPolicy
from utils import make_env, create_HADRL_PSN_file
from wrappers import ResetWithRealisticLoad, ResetWithLoadMixed

if __name__ == '__main__':
    psn_path = '../PSNs/waxman_20_servers.graphml'
    # nx.write_graphml(psn, psn_path)

    # create_HADRL_PSN_file(
    #     path=psn_path,
    #     n_CDCs=1,
    #     n_EDCs=1,
    #     n_servers_per_DC=(10, 6, 4),
    #     n_EDCs_per_CDC=1
    # )

    psn = reader.read_psn(psn_path)

    # training environment
    n_tr_envs = 20
    tr_nsprs_per_ep = 1
    tr_load = 0.9
    tr_time_limit = False
    tr_max_ep_steps = 1000
    tr_reset_load_class = ResetWithRealisticLoad
    # tr_reset_load_kwargs = dict(rand_load=True, rand_range=(0.3, 0.4))
    tr_reset_load_kwargs = dict(cpu_load=0.7)
    tr_env = make_vec_env(
        env_id=make_env,
        n_envs=n_tr_envs,
        env_kwargs=dict(
            psn_path=psn_path,
            time_limit=tr_time_limit,
            time_limit_kwargs=dict(max_episode_steps=tr_max_ep_steps),
            hadrl_nsprs=True,
            hadrl_nsprs_kwargs=dict(nsprs_per_ep=tr_nsprs_per_ep,
                                    vnfs_per_nspr=5,
                                    load=tr_load,
                                    always_one=True),
            reset_load_class=tr_reset_load_class,
            reset_load_kwargs=tr_reset_load_kwargs
        ),
    )

    # evaluation environment
    n_eval_envs = 4
    eval_nsprs_per_ep = 1
    eval_load = 0.9
    eval_time_limit = False
    eval_max_ep_steps = 1000
    eval_reset_load_class = ResetWithRealisticLoad
    # eval_reset_with_load_kwargs = dict(rand_load=True, rand_range=(0.3, 0.4))
    eval_reset_load_kwargs = dict(cpu_load=0.7)
    eval_env = make_vec_env(
        env_id=make_env,
        n_envs=n_eval_envs,
        env_kwargs=dict(
            psn_path=psn_path,
            time_limit=eval_time_limit,
            time_limit_kwargs=dict(max_episode_steps=eval_max_ep_steps),
            reset_load_class=eval_reset_load_class,
            reset_load_kwargs=eval_reset_load_kwargs,
            hadrl_nsprs=True,
            hadrl_nsprs_kwargs=dict(nsprs_per_ep=eval_nsprs_per_ep,
                                    vnfs_per_nspr=5,
                                    load=eval_load,
                                    always_one=True)
        ),
    )

    # model definition
    use_heuristic = True
    heu_kwargs = {'n_servers_to_sample': 4, 'heu_class': P2CLoadBalanceHeuristic,
                  'eta': 0.05, 'xi': 0.7, 'beta': 1.}
    policy = HADRLPolicy
    policy_kwargs = dict(psn=psn,
                         net_arch=[dict(pi=[128], vf=[128, 32])],
                         activation_fn=nn.Tanh,
                         servers_map_idx_id=tr_env.get_attr('servers_map_idx_id', 0)[0],
                         gcn_layers_dims=(20, 20, 20),
                         use_heuristic=use_heuristic,
                         heu_kwargs=heu_kwargs, )

    model = A2C(policy=policy, env=tr_env, verbose=2, device='cuda:0',
                learning_rate=0.0002,
                n_steps=1,  # ogni quanti step fare un update
                gamma=0.99,
                gae_lambda=0.92,
                ent_coef=0.01,
                # max_grad_norm=0.9,
                use_rms_prop=True,
                # tensorboard_log="../tb_logs/",
                policy_kwargs=policy_kwargs)

    # model = A2C(policy='MultiInputPolicy', env=tr_env, verbose=2, device='cuda:0',
    #             learning_rate=0.0005,
    #             n_steps=20,  # ogni quanti step fare un update
    #             gamma=0.99,
    #             ent_coef=0.01,
    #             gae_lambda=0.92,
    #             # max_grad_norm=0.9,
    #             use_rms_prop=True,
    #             tensorboard_log="../tb_logs_waxman-graph-inter-reward/",
    #             policy_kwargs=dict(
    #                 activation_fn=nn.Tanh,
    #                 net_arch=[128, dict(vf=[32])],
    #                 features_extractor_class=HADRLFeaturesExtractor,
    #                 features_extractor_kwargs=dict(
    #                     psn=psn,
    #                     activation_fn=nn.Tanh,
    #                     gcn_layers_dims=policy_kwargs['gcn_layers_dims'],
    #                 )
    #             ))

    print(model.policy)

    # define some training hyperparams
    tot_tr_steps = 20_000_000

    if tr_reset_load_class is not None:
        tr_load = tr_reset_load_kwargs.get('cpu_load', None)
    if eval_reset_load_class is not None:
        eval_load = eval_reset_load_kwargs.get('cpu_load', None)

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
    # wandb_run = wandb.init(
    #     project="Heuristic",
    #     dir="../",
    #     # name="Simpler HADRL-style PSN - branch main",
    #     config=config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     save_code=True,  # optional
    # )

    # training callbacks
    list_of_callbacks = [
        # AcceptanceRatioByStepsCallback(env=tr_env, name="Acceptance ratio (by steps)",
        #                                steps_per_tr_phase=500, verbose=2),

        AcceptanceRatioByNSPRsCallback(env=tr_env, name="Train acceptance ratio (by NSPRs)",
                                       nsprs_per_tr_phase=1000, verbose=2),

        HParamCallback(tr_env.num_envs, eval_env.num_envs, tr_nsprs_per_ep,
                       tr_load,
                       tr_max_ep_steps=tr_max_ep_steps if tr_time_limit else None,
                       eval_nsprs_per_ep=eval_nsprs_per_ep,
                       eval_psn_load=eval_load,
                       eval_max_ep_steps=eval_max_ep_steps if eval_time_limit else None,
                       use_heuristic=use_heuristic, heu_kwargs=heu_kwargs, ),

        # WandbCallback(model_save_path=f"../models/{wandb_run.id}",
        #               verbose=2,
        #               model_save_freq=10_000),

        EvalCallback(eval_env=eval_env, n_eval_episodes=1000, warn=True,
                     eval_freq=5_000, deterministic=True, verbose=2,
                     callback_after_eval=AcceptanceRatioByNSPRsCallback(
                         env=eval_env,
                         name="Eval acceptance ratio (by NSPRs)",
                         nsprs_per_tr_phase=1,  # must be 1 for eval (default value)
                         verbose=2
                     )),

        PSNLoadCallback(env=tr_env, freq=50, verbose=1),

        SeenNSPRsCallback(env=tr_env, freq=50, verbose=1),
    ]

    # model training
    model.learn(total_timesteps=tot_tr_steps,
                log_interval=10,
                # tb_log_name="A2C_Adam",
                callback=list_of_callbacks)

    # wandb_run.finish()
