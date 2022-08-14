"""
Implementation of a space consisting of finitely many elements.

DISCLAIMER:
This file is taken and slightly modified from the Discrete space of OpenAI gym release 0.25.1.

stable-baselines3-1.5.0 requires gym==0.21, since they introduced breaking changes in 0.22.
In this project, it is required to have a discrete space with the 'start' attribute, which
was introduced only in later versions of gym, therefore a custom space
(similar to later versions of the Discrete space in gym) is needed.
"""

from typing import Optional, Union

import numpy as np

from gym.spaces.space import Space
from gym.utils import seeding


class DiscreteWithNegatives(Space):
    r"""A space consisting of finitely many elements.

    This class represents a finite subset of integers, more specifically a set of the form :math:`\{ a, a+1, \dots, a+n-1 \}`.

    Example::

        >>> DiscreteWithNegatives(2)            # {0, 1}
        >>> DiscreteWithNegatives(3, start=-1)  # {-1, 0, 1}
    """

    def __init__(
        self,
        n: int,
        seed: Optional[int] = None,
        start: int = 0,
    ):
        r"""Constructor of :class:`Discrete` space.

        This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

        Args:
            n (int): The number of elements of this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
            start (int): The smallest element of this space.
        """
        assert isinstance(n, (int, np.integer))
        assert n > 0, "n (counts) have to be positive"
        assert isinstance(start, (int, np.integer))
        self.n = int(n)
        self.start = int(start)
        super().__init__((), np.int64, seed)

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape `(n,)` and dtype `np.int8` where `1` represents valid actions and `0` invalid / infeasible actions.
                If there are no possible actions (i.e. `np.all(mask == 0)`) then `space.start` will be returned.

        Returns:
            A sampled integer from the space
        """
        if mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            assert mask.shape == (
                self.n,
            ), f"The expected shape of the mask is {(self.n,)}, actual shape: {mask.shape}"
            valid_action_mask = mask == 1
            assert np.all(
                np.logical_or(mask == 0, valid_action_mask)
            ), f"All values of a mask should be 0 or 1, actual values: {mask}"
            if np.any(valid_action_mask):
                return int(
                    self.start + self.np_random.choice(np.where(valid_action_mask)[0])
                )
            else:
                return self.start

        return int(self.start + self.np_random.randint(self.n))

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)  # type: ignore
        else:
            return False
        return self.start <= as_int < self.start + self.n

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return "DiscreteWithNegatives(%d, start=%d)" % (self.n, self.start)
        return "DiscreteWithNegatives(%d)" % self.n

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, DiscreteWithNegatives)
            and self.n == other.n
            and self.start == other.start
        )

    def __setstate__(self, state):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        super().__setstate__(state)

        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = 0

        # Update our state
        self.__dict__.update(state)
