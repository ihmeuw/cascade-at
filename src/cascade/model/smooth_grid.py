from math import nan, inf

import numpy as np
import pandas as pd

from cascade.dismod.constants import PriorKindEnum
from cascade.model.age_time_grid import AgeTimeGrid
from cascade.model.priors import prior_distribution


class _PriorView(AgeTimeGrid):
    """Slices to access priors with Distribution objects.
    Each PriorView has one mulstd, corresponding with its kind.
    """
    def __init__(self, kind, ages, times):
        self._kind = kind
        super().__init__(
            (ages, times),
            columns=["density", "mean", "std", "lower", "upper", "eta", "nu", "name"])
        # Let the base class make extra mulstds and delete them.
        for del_kind in PriorKindEnum:
            if del_kind.name != kind:
                del self.mulstd[del_kind.name]

    def age_time_diff(self):
        yield from zip(
            np.repeat(self.ages, len(self.times)),
            np.tile(self.times, len(self.ages)),
            np.repeat(np.ediff1d(self.ages, to_end=inf), len(self.times)),
            np.tile(np.ediff1d(self.times, to_end=inf), len(self.ages)),
        )

    def __getitem__(self, at_slice):
        return prior_distribution(super().__getitem__(at_slice).iloc[0])

    def __setitem__(self, at_slice, value):
        """
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
    def __init__(self, age_time_grid):
        """
        The Smooth Grid is a set of priors on an age-time grid.

        Args:
            age_time_grid (Tuple(set,set)): The supporting grid.
        """
        self.ages = np.sort(np.array(age_time_grid[0], dtype=np.float))
        self.times = np.sort(np.array(age_time_grid[1], dtype=np.float))
        self._view = dict()
        for create_view in PriorKindEnum:
            self._view[create_view.name] = _PriorView(create_view.name, self.ages, self.times)

    def __len__(self):
        return sum(len(v) for v in self._view.values())

    def __str__(self):
        return f"SmoothGrid({len(self.ages), len(self.times)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._view == other._view

    def age_time(self):
        yield from zip(np.repeat(self.ages, len(self.times)), np.tile(self.times, len(self.ages)))

    def age_time_diff(self):
        yield from zip(
            np.repeat(self.ages, len(self.times)),
            np.tile(self.times, len(self.ages)),
            np.repeat(np.ediff1d(self.ages, to_end=inf), len(self.times)),
            np.tile(np.ediff1d(self.times, to_end=inf), len(self.ages)),
        )

    @property
    def value(self):
        return self._view["value"]

    @property
    def dage(self):
        return self._view["dage"]

    @property
    def dtime(self):
        return self._view["dtime"]

    @property
    def priors(self):
        """All priors in one dataframe."""
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
    smooth_grid = SmoothGrid((var.ages, var.times))
    if strictly_positive:
        smooth_grid.value.grid.loc[:, ["density", "mean", "lower", "upper"]] = [
            "uniform", 1e-2, 1e-9, 5
        ]
    else:
        smooth_grid.value.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -5, 5, 0]
    smooth_grid.dage.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -5, 5, 0]
    smooth_grid.dtime.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -5, 5, 0]
    return smooth_grid
