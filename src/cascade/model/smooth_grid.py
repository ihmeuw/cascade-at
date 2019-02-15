from math import nan, inf

import numpy as np
import pandas as pd

from cascade.dismod.constants import PriorKindEnum
from cascade.model.age_time_grid import AgeTimeGrid
from cascade.model.priors import prior_distribution
from cascade.model.var import Var


class _PriorGrid(AgeTimeGrid):
    """Slices to access priors with Distribution objects.
    Each PriorView has one mulstd, corresponding with its kind.
    """
    def __init__(self, kind, ages, times):
        self._kind = kind
        super().__init__(
            ages, times,
            columns=["density", "mean", "std", "lower", "upper", "eta", "nu", "name"])
        # Let the base class make extra mulstds and delete them.
        for del_kind in PriorKindEnum:
            if del_kind.name != kind:
                del self._mulstd[del_kind.name]

    def age_time_diff(self):
        """Iterate over (age, time) in the grid."""
        yield from zip(
            np.repeat(self.ages, len(self.times)),
            np.tile(self.times, len(self.ages)),
            np.repeat(np.ediff1d(self.ages, to_end=inf), len(self.times)),
            np.tile(np.ediff1d(self.times, to_end=inf), len(self.ages)),
        )

    @property
    def mulstd_prior(self):
        """Standard deviation multiplier as a Prior. Returns a Prior
        or None, if the prior is not defined."""
        # The base class, AgeTimeGrid, has a dictionary of three mulstds.
        # The prior grid uses only one of them.
        return prior_distribution(self._mulstd[self._kind].iloc[0])

    @mulstd_prior.setter
    def mulstd_prior(self, value):
        """Erase a mulstd by setting it to None."""
        if value is not None:
            to_set = value.parameters()
            to_assign = [to_set[setp] if setp in to_set else nan for setp in self.columns]
            self._mulstd[self._kind].loc[:, self.columns] = to_assign
        else:
            self._mulstd[self._kind].loc[:, self.columns] = [None, 0, .1, -inf, inf, nan, nan, None]

    def __getitem__(self, at_slice):
        return prior_distribution(super().__getitem__(at_slice).iloc[0])

    def __setitem__(self, at_slice, value):
        """
        These can't be erased because every grid point gets a prior.

        Args:
            at_slice (slice, slice): What to change, as integer offset into ages and times.
            value (priors.Prior): The prior to set, containing dictionary of
                                  parameters.
        """
        to_set = value.parameters()
        super().__setitem__(at_slice, [to_set[setp] if setp in to_set else nan for setp in self.columns])

    def apply(self, transform):
        for idx, row in self.grid.loc[:, self.columns + ["age", "time"]].iterrows():
            new_distribution = transform(row.age, row.time, prior_distribution(row))
            to_set = new_distribution.parameters()
            self.grid.iloc[idx, self.columns] = [
                to_set[setp] if setp in to_set else nan for setp in self.columns]


class SmoothGrid:
    def __init__(self, ages, times):
        """
        The Smooth Grid is a set of priors on an age-time grid.

        Args:
            age_time_grid (Tuple(set,set)): The supporting grid.
        """
        self.ages = np.sort(np.array(ages, dtype=np.float))
        self.times = np.sort(np.array(times, dtype=np.float))
        self._view = dict()
        for create_view in PriorKindEnum:
            self._view[create_view.name] = _PriorGrid(create_view.name, self.ages, self.times)

    def variable_count(self):
        """A Dismod-AT fit solves for model variables. This counts how many
        model variables are defined by this SmoothGrid, which indicates how
        much this SmoothGrid contributes to the size of the problem."""
        return sum(v.variable_count() for v in self._view.values())

    def var_from_mean(self):
        """Given a prior grid, create a Var from the mean of the value priors.

        Returns:
            Var: A new Var object with the same ages and times and value
            equal to the mean.
        """
        var = Var(self.ages, self.times)
        for age, time in self.age_time():
            var[age, time] = self.value[age, time].mean
        return var

    def __len__(self):
        return self.variable_count()

    def __str__(self):
        return f"SmoothGrid({len(self.ages), len(self.times)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._view == other._view

    def age_time(self):
        """Iterate over (age, time) in the grid."""
        yield from zip(np.repeat(self.ages, len(self.times)), np.tile(self.times, len(self.ages)))

    def age_time_diff(self):
        """Iterate over (age, time, forward difference in age,
        forward difference in time). The last differences will be inf."""
        yield from zip(
            np.repeat(self.ages, len(self.times)),
            np.tile(self.times, len(self.ages)),
            np.repeat(np.ediff1d(self.ages, to_end=inf), len(self.times)),
            np.tile(np.ediff1d(self.times, to_end=inf), len(self.ages)),
        )

    @property
    def value(self):
        """Grid of value priors."""
        return self._view["value"]

    @property
    def dage(self):
        """Grid of priors on differences in age."""
        return self._view["dage"]

    @property
    def dtime(self):
        """Grid of priors on differences in time."""
        return self._view["dtime"]

    @property
    def priors(self):
        """All priors in one dataframe. Used for serialization."""
        total = list()
        for kind, view in self._view.items():
            total.append(view.grid.assign(kind=kind))
            total.append(view.mulstd[kind].assign(kind=kind))
        return pd.concat(total).reset_index(drop=True)


def uninformative_grid_from_var(var, strictly_positive):
    """
    Create a smooth grid with priors that are Uniform and
    impossibly large, in the same shape as a Var.

    Args:
        var (Var): A single var grid.
        strictly_positive (bool): Whether the value prior is positive.

    Returns:
        SmoothGrid: A single smooth grid with Uniform distributions.
    """
    smooth_grid = SmoothGrid(var.ages, var.times)
    if strictly_positive:
        smooth_grid.value.grid.loc[:, ["density", "mean", "lower", "upper"]] = [
            "uniform", 1e-2, 1e-9, 5
        ]
    else:
        smooth_grid.value.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -inf, inf, 0]
    smooth_grid.dage.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -inf, inf, 0]
    smooth_grid.dtime.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -inf, inf, 0]
    return smooth_grid
