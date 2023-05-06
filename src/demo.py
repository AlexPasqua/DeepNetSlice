from trainer import Trainer
from wrappers.reset_with_load import ResetWithRealisticLoad


if __name__ == '__main__':
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
    )
