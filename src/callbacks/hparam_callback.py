from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    def __init__(
            self,
            n_tr_envs: int = None,
            n_eval_envs: int = None,
            tr_nsprs_per_ep: int = None,
            tr_psn_load: float = None,
            tr_max_ep_steps: int = None,
            eval_nsprs_per_ep: int = None,
            eval_psn_load: float = None,
            eval_max_ep_steps: int = None,
            vnfs_per_nsprs: int = None,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
    ):
        """
        Saves the hyperparameters and metrics at the start of the training,
        and logs them to TensorBoard.

        :param n_tr_envs: number of training environments
        """
        super().__init__()
        self.n_tr_envs = n_tr_envs
        self.n_eval_envs = n_eval_envs
        self.tr_nsprs_per_ep = tr_nsprs_per_ep
        self.tr_psn_load = tr_psn_load
        self.tr_max_ep_steps = tr_max_ep_steps
        self.eval_nsprs_per_ep = eval_nsprs_per_ep
        self.eval_psn_load = eval_psn_load
        self.eval_max_ep_steps = eval_max_ep_steps
        self.vnfs_per_nspr = vnfs_per_nsprs
        self.use_heuristic = use_heuristic
        self.heu_kwargs = heu_kwargs if heu_kwargs is not None else {}
        if 'heu_class' in self.heu_kwargs:
            self.heu_class = self.heu_kwargs['heu_class'](None, None, None).__class__.__name__
        else:
            self.heu_class = None

    def _on_training_start(self) -> None:
        try:
            gcn_layers_dims = str(self.model.policy.features_extractor.gcn_layers_dims)
        except AttributeError:
            gcn_layers_dims = str(self.model.policy.gcn_layers_dims)

        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "n training envs": self.n_tr_envs,
            "n eval envs": self.n_eval_envs,
            "n steps before update": self.model.n_steps,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "entropy coefficient": self.model.ent_coef,
            "NSPRs per training episode": self.tr_nsprs_per_ep,
            "max steps per training episode": self.tr_max_ep_steps,
            "PSN load (training)": self.tr_psn_load,
            "NSPRs per eval episode": self.eval_nsprs_per_ep,
            "PSN load (eval)": self.eval_psn_load,
            "max steps per eval episode": self.eval_max_ep_steps,
            "VNFs/NSPR": self.vnfs_per_nspr,
            "GCN layers dimensions": gcn_layers_dims,
            "Use heuristic": self.use_heuristic,
            "Heuristic class": self.heu_class,
            "heu's num sampled servers": self.heu_kwargs.get("n_servers_to_sample", None),
            "heu's eta": self.heu_kwargs.get("eta", None),
            "heu's xi": self.heu_kwargs.get("xi", None),
            "heu's beta": self.heu_kwargs.get("beta", None),
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorboard will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "Acceptance ratio": 0,
            "Eval acceptance ratio": 0,
            "eval/mean_reward": 0,
            "rollout/ep_rew_mean": 0,
            "train/entropy_loss": 0,
            "train/policy_loss": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
