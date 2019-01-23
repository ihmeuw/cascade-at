from collections import UserDict
from os import linesep


class DismodGroups(UserDict):
    """Dismod-AT documentation discusses five kinds of groups of
    model variables (https://bradbell.github.io/dismod_at/doc/model_variables.htm).
    This class represents that grouping as a set of dictionaries, where the
    values can either be SmoothGrids or RandomFields or whatever is classified
    according to groups of model variables.

     * Rate key is rate as a string (iota, rho, chi, omega, pini)
     * Random effect key is tuple (rate, location_id), where None is all.
     * Alpha key is (covariate, rate), both as strings.
     * Beta key is (covariate, integrand), both as strings.
     * Gamma key is (covariate, integrand), both as strings.

    """
    GROUPS = ["rate", "random_effect", "alpha", "beta", "gamma"]

    def __init__(self):
        super().__init__({k: dict() for k in self.GROUPS})
        self._frozen = True

    def __getattr__(self, item):
        if item in self.GROUPS:
            return self.data[item]
        else:
            raise AttributeError(f"{item} is not an attribute")

    def __setattr__(self, item, value):
        if item in self.GROUPS and self.__dict__.get("_frozen", False):
            raise AttributeError(f"Cannot set attribute")
        else:
            self.__dict__[item] = value

    def __setitem__(self, key, item):
        """
        This keeps us from treating this class as a dictionary by accident.
        """
        if self.__dict__.get("_frozen", False):
            raise ValueError("Cannot set property on a DismodGroups object.")
        else:
            super().__setitem__(key, item)

    def variable_count(self):
        """Sum of lengths of values in the container."""
        total = 0
        for group in self.values():
            for container in group.values():
                total += container.variable_count()
        return total

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        for group_name, group in self.data.items():
            if group != other.data[group_name]:
                return False

    def __str__(self):
        message = list()
        for group_name, group in self.items():
            message.append(f"{group_name}")
            for key, value in group.items():
                message.append(f"  {key}: {value}")
        return linesep.join(message)
