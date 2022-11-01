import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AcceptanceRatioCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    It logs the acceptance ratio on Tensorboard.

    :param env: environment
    :param name: name of the metric to log
    :param steps_per_tr_phase: number of steps that define a training phase.
        The acceptance ratio is logged once per training phase.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
            self,
            env: gym.Env,
            name: str = "Acceptance ratio",
            steps_per_tr_phase: int = 1,
            verbose=0
    ):
        super(AcceptanceRatioCallback, self).__init__(verbose)
        self.env = env
        self.name = name
        self.steps_per_tr_phase = steps_per_tr_phase
        self.tot_to_subtract = None
        self.accepted_to_subtract = None
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.steps_per_tr_phase == 0:
            accepted_nsprs_per_env = np.array(self.env.get_attr("accepted_nsprs"), dtype=np.float32)
            tot_nsprs_per_env = np.array(self.env.get_attr("tot_seen_nsprs"), dtype=np.float32)
            if self.tot_to_subtract is None:    # or self.accepted_to_subtract is None, either way
                self.tot_to_subtract = np.zeros_like(tot_nsprs_per_env)
                self.accepted_to_subtract = np.zeros_like(accepted_nsprs_per_env)
            accepted_nsprs_per_env -= self.accepted_to_subtract
            tot_nsprs_per_env -= self.tot_to_subtract
            accept_ratio_per_env = np.divide(accepted_nsprs_per_env,
                                             tot_nsprs_per_env,
                                             out=np.zeros_like(tot_nsprs_per_env),
                                             where=tot_nsprs_per_env != 0)
            overall_accept_ratio = np.mean(accept_ratio_per_env)
            self.logger.record(self.name, overall_accept_ratio)
            self.tot_to_subtract = tot_nsprs_per_env
            self.accepted_to_subtract = accepted_nsprs_per_env
        return True
