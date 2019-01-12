from collections import UserDict
from os import linesep


class DismodGroups(UserDict):
    """Dismod-AT documentation discusses five kinds of groups of
    model variables (https://bradbell.github.io/dismod_at/doc/model_variables.htm).
    This class represents that grouping as a set of dictionaries, where the
    values can either be SmoothGrids or RandomFields or whatever is classified
    according to groups of model variables.
    """
    GROUPS = ["rate", "random_effect", "alpha", "beta", "gamma"]

    def __init__(self):
        # Key is the rate as a string.
        self.rate = dict()
        # Key is tuple (rate, location_id)  # location_id=None means no nslist.
        self.random_effect = dict()
        # Key is (covariate, rate), both as strings.
        self.alpha = dict()
        # Key is (covariate, integrand), both as strings.
        self.beta = dict()
        # Key is (covariate, integrand), both as strings.
        self.gamma = dict()
        self._frozen = False
        super().__init__({k: getattr(self, k) for k in self.GROUPS})
        self._frozen = True

    def __setitem__(self, key, item):
        """
        This keeps us from treating this class as a dictionary by accident.
        """
        if self._frozen:
            raise ValueError("Cannot set property on a DismodGroups object.")
        else:
            super().__setitem__(key, item)

    def count(self):
        """Sum of lengths of values in the container."""
        total = 0
        for group in self.values():
            for container in group.values():
                total += len(container)
        return total

    def __str__(self):
        message = list()
        for group_name, group in self.items():
            message.append(f"{group_name}")
            for key, value in group.items():
                message.append(f"  {key}: {value}")
        return linesep.join(message)