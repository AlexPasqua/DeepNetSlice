from callbacks.acceptance_ratio_callbacks import AcceptanceRatioByNSPRsCallback
from callbacks.hparam_callback import HParamCallback
from callbacks.psn_load_callback import PSNLoadCallback
from callbacks.seen_nsprs_callback import SeenNSPRsCallback
from trainer import Trainer
from wrappers.reset_with_load import ResetWithRealisticLoad
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback


if __name__ == '__main__':
    # create trainer object.
    # It creates the model and the training and evaluation environments.
    trainer = Trainer(
        psn_path="../PSNs/hadrl_1-16_5-10_15-4.graphml",
        n_tr_envs=20,
        load_perc=0.8,
        time_limit=False,
        max_ep_steps=1000,
        reset_load_class=ResetWithRealisticLoad,
        generate_nsprs=True,
        nsprs_per_ep=1,
        vnfs_per_nspr=5,
        always_one=True,
        seed=12,
        tensorboard_log="../tensorboard",
        create_eval_env=True
    )
    tr_env = trainer.tr_env
    eval_env = trainer.eval_env

    # training callbacks
    list_of_callbacks = [
        AcceptanceRatioByNSPRsCallback(
            env=tr_env,
            name="Train acceptance ratio (by NSPRs)",
            nsprs_per_tr_phase=1000,
            verbose=2
        ),

        EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=1000,
            warn=True,
            eval_freq=5_000,
            deterministic=True,
            verbose=2,
            callback_after_eval=AcceptanceRatioByNSPRsCallback(
                env=eval_env,
                name="Eval acceptance ratio (by NSPRs)",
                nsprs_per_tr_phase=1,  # must be 1 for eval (default value)
                verbose=2
            )
        ),

        PSNLoadCallback(env=tr_env, freq=500, verbose=1),

        # SeenNSPRsCallback(env=tr_env, freq=100, verbose=1),
    ]

    trainer.train(
        tot_steps=10_000_000,
        callbacks=list_of_callbacks,
    )
