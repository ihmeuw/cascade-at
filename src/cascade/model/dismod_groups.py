from collections import UserDict
from itertools import chain
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
            raise AttributeError(f"{item} must be an attribute in {self.GROUPS}")

    def __setattr__(self, item, value):
        if item in self.GROUPS and self.__dict__.get("_frozen", False):
            raise AttributeError(f"Cannot set attribute {item} because it is not writable.")
        else:
            object.__setattr__(self, item, value)

    def __setitem__(self, key, item):
        """
        This keeps us from treating this class as a dictionary by accident.
        """
        if self.__dict__.get("_frozen", False):
            raise ValueError(f"Cannot set property {key} on a DismodGroups object.")
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

    def check_alignment(self, other):
        """Check whether and where two DismodGroups are misaligned."""
        one_not_other = list()
        for group_name, group in self.items():
            if group_name != "random_effect":
                message = self._compare_keys(group.keys(), other[group_name].keys(), group_name)
                one_not_other.extend(message)
            else:
                re_rates = {rk[0] for rk in chain(group.keys(), other[group_name].keys())}
                for re_rate in re_rates:
                    a_re_key = [ak for ak in group.keys() if ak[0] == re_rate]
                    b_re_key = [bk for bk in other[group_name].keys() if bk[0] == re_rate]
                    single_effect = len(a_re_key) == 1 or len(b_re_key) == 1
                    all_effects = a_re_key[0][1] is None or b_re_key[0][1] is None
                    if single_effect and all_effects:
                        a_re_key = [k[0] for k in a_re_key]
                        b_re_key = [k[0] for k in b_re_key]
                    message = self._compare_keys(a_re_key, b_re_key, group_name)
                    one_not_other.extend(message)

        return one_not_other

    def _compare_keys(self, a_keys, b_keys, group_name):
        a_keys = set(a_keys)
        b_keys = set(b_keys)
        message = []
        if a_keys - b_keys:
            message = [f"left {group_name} has {a_keys - b_keys}"]
        if b_keys - a_keys:
            message = [f"right {group_name} has {b_keys - a_keys}"]
        return message
