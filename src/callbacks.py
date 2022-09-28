from stable_baselines3.common.callbacks import BaseCallback


class AcceptanceRatioCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    It logs the acceptance ratio on Tensorboard.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, name: str = "Acceptance ratio", verbose=0):
        super(AcceptanceRatioCallback, self).__init__(verbose)
        self.name = name
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
        list_of_totals = self.training_env.get_attr("tot_nsprs")
        list_of_accepted = self.training_env.get_attr("accepted_nsprs")
        local_accept_ratios = []
        for i, tot in enumerate(list_of_totals):
            if tot > 0:
                cur_accept_ratio = list_of_accepted[i] / tot
                local_accept_ratios.append(cur_accept_ratio)
        n_ratios = len(local_accept_ratios)
        if n_ratios > 0:
            overall_accept_ratio = sum(local_accept_ratios) / n_ratios
            self.logger.record(self.name, overall_accept_ratio)
        return True
