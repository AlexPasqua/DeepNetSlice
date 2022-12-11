from stable_baselines3.common.callbacks import BaseCallback
import gym
import numpy as np


class SeenNSPRsCallback(BaseCallback):
    """
    Class for logging the number of seen NSPRs so far.

    It logs the average number of seen NSPRs for each environment.
    The average is chosen, instead of the sum, because the loss is based on the
    average of the "values" in the various steps:
        - policy_loss = -(advantages * log_prob).mean()
        - value_loss = F.mse_loss(rollout_data.returns, values)
        - entropy_loss = -th.mean(entropy)
    If there are multiple parallel envs, the "values" of each env are flattened, 
    and again the average is computed for the loss.
    Therefore, we don't have more updates if we have more envs, just more precise.
    If 2 envs have seen 10 NSPRs, it's not like an env has seen 20 (in terms of updates and steps).

    :param env: environment
    :param freq: logging frequency (in number of steps)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
            self,
            env: gym.Env,
            freq: int = 1,
            verbose: int = 0
    ):
        super().__init__(verbose)
        self.env = env
        self.freq = freq

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            # log the number of seen NSPRs
            seen_nsprs_per_env = self.env.get_attr('tot_seen_nsprs')
            # why the mean and not the sum, you ask? Read the docstring of the class
            avg_seen_nsprs = int(round(np.mean(seen_nsprs_per_env)))
            self.logger.record("Avg seen NSPRs per env", avg_seen_nsprs)
            if self.verbose > 0:
                print(f"Average seen NSPRs per env: {avg_seen_nsprs}")
        return True