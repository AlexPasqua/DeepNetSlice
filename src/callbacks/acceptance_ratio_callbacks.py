import warnings
from queue import Queue
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class AcceptanceRatioByStepsCallback(BaseCallback):
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
        super(AcceptanceRatioByStepsCallback, self).__init__(verbose)
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


class AcceptanceRatioByNSPRsCallback(BaseCallback):
    """
   A custom callback that derives from ``BaseCallback``.
   It logs the acceptance ratio on Tensorboard.

   Note: it works only with non-vectorized environment or with a vectorized one
   containing only 1 environment.

   :param env: environment
   :param name: name of the metric to log
   :param nsprs_per_tr_phase: number of NSPRs that define a training phase.
       The acceptance ratio is logged once per training phase.
   :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
   """
    def __init__(
            self,
            env: gym.Env,
            name: str = "Acceptance ratio",
            nsprs_per_tr_phase: int = 1000,
            verbose=0
    ):
        super().__init__(verbose)
        self.env = env
        self.name = name
        self.nsprs_per_tr_phase = nsprs_per_tr_phase
        # num of seen NSPRs to subtract form the total number of seen NSPRs (per env)
        self.seen_to_subtract = [0] * env.num_envs
        # num of accepted NSPRs to subtract form the total number of accepted NSPRs (per env)
        self.accepted_to_subtract = [0] * env.num_envs
        # num of seen NSPRs last considered for logging (per env),
        # used to ensure it loggs once per training phase
        self.last_seen = [0] * env.num_envs
        # num of accepted NSPRs during this training phase (per env)
        self.accepted_this_training_phase = [0] * env.num_envs
        # num of NSPRs seen during this training phase (per env)
        self.seen_this_training_phase = [0] * env.num_envs
        # acceptance ratio of each env
        self.acceptance_ratios = [Queue() for _ in range(env.num_envs)]
        # once an env is ready for logging, its cell is increased by 1,
        # and it is decreased by 1 when the acceptance ratio is logged
        self.ready_envs = np.zeros(shape=env.num_envs, dtype=int)
        if isinstance(self.env, VecEnv) and self.env.num_envs > 1:
            warnings.warn("The env is vectorized, only the first env instance "
                          "will be used for the acceptance ratio by NSPRs.")

    def _on_step(self) -> bool:
        if isinstance(self.env, VecEnv):
            seen_nsprs = self.env.get_attr('tot_seen_nsprs')
            accepted_nsprs = self.env.get_attr('accepted_nsprs')
        else:
            seen_nsprs = [self.env.tot_seen_nsprs]
            accepted_nsprs = [self.env.accepted_nsprs]
        
        for env_idx in range(self.env.num_envs):
            if seen_nsprs[env_idx] > self.last_seen[env_idx] and seen_nsprs[env_idx] % self.nsprs_per_tr_phase == 0:
                self.ready_envs[env_idx] += 1
                self.last_seen[env_idx] = seen_nsprs[env_idx]
                # NSPRs seen and accepted in this training phase
                seen_this_tr_phase = seen_nsprs[env_idx] - self.seen_to_subtract[env_idx]
                accepted_this_tr_phase = accepted_nsprs[env_idx] - self.accepted_to_subtract[env_idx]
                # update how much to subtract to get the quantities for next tr phase
                self.seen_to_subtract[env_idx] = seen_nsprs[env_idx]
                self.accepted_to_subtract[env_idx] = accepted_nsprs[env_idx]
                # compute acceptance ratio
                try:
                    self.acceptance_ratios[env_idx].put(accepted_this_tr_phase / seen_this_tr_phase)
                except ZeroDivisionError:
                    self.acceptance_ratios[env_idx].put(0.)
        
        if all(self.ready_envs):
            ratios = [self.acceptance_ratios[env_idx].get() for env_idx in range(self.env.num_envs)]
            self.logger.record(self.name, np.mean(ratios))
            self.ready_envs -= 1
        
        return True
