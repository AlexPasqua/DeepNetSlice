import copy
from typing import List, Optional, Type

import gym
import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from torch import nn
from wandb.integration.sb3 import WandbCallback

import reader
from callbacks.acceptance_ratio_callbacks import AcceptanceRatioByNSPRsCallback
from callbacks.hparam_callback import HParamCallback
from callbacks.psn_load_callback import PSNLoadCallback
from callbacks.seen_nsprs_callback import SeenNSPRsCallback
from policies.features_extractors.hadrl_features_extractor import \
    GCNsFeaturesExtractor
from utils import make_env


class Trainer:
    def __init__(
        self,
        psn_path: str,
        n_tr_envs: int,
        load_perc: float,
        time_limit: bool,
        max_ep_steps: int,
        tensorboard_log: str,
        create_eval_env: bool = False,
        reset_load_class: Optional[gym.Wrapper] = None,
        reset_load_kwargs: dict = dict(cpu_load=0.8),
        # reset_load_kwargs: dict = dict(rand_load=True, rand_range=(0., 1.)),
        placement_state: bool = True,
        accumulate_rew: bool = True,
        discount_acc_rew: bool = True,
        dynamic_connectivity: int = False,
        dynamic_connectivity_kwargs: dict = dict(link_bw=10_000),
        generate_nsprs: bool = True,
        nsprs_per_ep: int = 1,
        vnfs_per_nspr: int = 5,
        always_one: bool = True,
        seed: Optional[int] = None,
        net_arch: dict = dict(pi=[256, 128], vf=[256, 128, 32]),
        activation_fn: Type[nn.Module] = nn.Tanh,
        gcn_layers_dims: tuple = (20, 20, 20),
        device: str = 'cuda:0',
        lr: float = 0.0002,
        n_steps: int = 1,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        gae_lambda: float = 0.92,
        # eval_load: Optional[float] = None,
    ):
        # checks on argumetns
        assert n_tr_envs > 0
        assert 0. <= load_perc < 1., "Training load must be a percentage between 0 and 1"

        # save some attributes
        self.nsprs_per_ep = nsprs_per_ep
        self.max_ep_steps = max_ep_steps
        self.time_limit = time_limit
        self.placement_state = placement_state

        # read PSN file
        psn = reader.read_psn(psn_path)

        # create trainin environment
        self.tr_env = make_vec_env(
            env_id=make_env,
            n_envs=n_tr_envs,
            env_kwargs=dict(
                psn_path=psn_path,
                base_env_kwargs=dict(
                    accumulate_reward=accumulate_rew,
                    discount_acc_rew=discount_acc_rew,
                ),
                time_limit=time_limit,
                time_limit_kwargs=dict(max_episode_steps=max_ep_steps),
                generate_nsprs=generate_nsprs,
                nsprs_gen_kwargs=dict(
                    nsprs_per_ep=nsprs_per_ep,
                    vnfs_per_nspr=vnfs_per_nspr,
                    load=load_perc,
                    always_one=always_one
                ),
                reset_load_class=reset_load_class,
                reset_load_kwargs=reset_load_kwargs,
                placement_state=placement_state,
                dynamic_connectivity=dynamic_connectivity,
                dynamic_connectivity_kwargs=dynamic_connectivity_kwargs
            ),
            seed=seed,
        )

        # create evaluation environment
        if create_eval_env:
            self.eval_env = copy.deepcopy(self.tr_env)

        # create the model
        self.model = A2C(policy='MultiInputPolicy', env=self.tr_env, verbose=2, device=device,
                    learning_rate=lr,
                    n_steps=n_steps,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    gae_lambda=gae_lambda,
                    seed=seed,
                    use_rms_prop=True,
                    tensorboard_log=tensorboard_log,
                    policy_kwargs=dict(
                        activation_fn=activation_fn,
                        net_arch=net_arch,
                        features_extractor_class=GCNsFeaturesExtractor,
                        share_features_extractor=False,
                        features_extractor_kwargs=dict(
                            psn=psn,
                            activation_fn=nn.ReLU,
                            gcn_layers_dims=gcn_layers_dims,
                        )
                    ))
        print(self.model.policy)

        # wandb config
        if reset_load_kwargs.get('rand_load', False):
            load_range = reset_load_kwargs.get('rand_range', (0., 1.))
            self.tr_load = 'random ' + str(load_range)
        else:
            self.tr_load = reset_load_kwargs.get('cpu_load', 0.8)
        # eval_load = eval_load if eval_load is not None else self.tr_load
        self.wandb_config = {
            "n tr envs": n_tr_envs,
            "NSPRs per training ep": nsprs_per_ep,
            "max steps per tr ep": max_ep_steps if time_limit else None,
            "PSN load (tr)": self.tr_load,
            # "PSN load (eval)": eval_load,
            "GCNs layers dims": gcn_layers_dims,
            "mpl_extractor arch": net_arch,
            "use placement state": placement_state,
            "accumulate reward": accumulate_rew,
            "discount acceptance reward": discount_acc_rew,
            "dynamic connectivity": dynamic_connectivity,
            "dynamic load range": "0-0.9",
        }

    def train(
            self,
            tot_steps: int,
            log_interval: int = 10,
            wandb: bool = False,
            callbacks: List[BaseCallback] = [],
    ):
        # wandb things
        self.wandb_config["total training steps"] = tot_steps
        if wandb:
            # init wandb run
            wandb_run = wandb.init(
                project="Same or different activations",
                dir="../",
                name="SAME (ReLU) (non-shared f.e.) (wax50, load 0.8, small GCNs)",
                config=self.wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )
            # add wandb callback
            callbacks.append(
                WandbCallback(
                    model_save_path=f"../models/{wandb_run.id}",
                    verbose=2,
                    model_save_freq=10_000
                )
            )
        
        # add callback for hyperparameters logging
        callbacks.append(
            HParamCallback(
                self.tr_env.num_envs,
                self.eval_env.num_envs,
                self.nsprs_per_ep,
                self.tr_load,
                tr_max_ep_steps=self.max_ep_steps if self.time_limit else None,
                use_placement_state=self.placement_state,
            ),
        )
        
        # model training
        self.model.learn(
            total_timesteps=tot_steps,
            log_interval=log_interval,
            callback=callbacks
        )

        if wandb:
            wandb_run.finish()
