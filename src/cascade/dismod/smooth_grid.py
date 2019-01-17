"""
The model is a collection of random fields plus some specification
for the solver.

All the kinds of objects:

 * Random fields in a model.
   * age-time grid
   * priors
     * prior distribution and prior parameters on a grid
     * prior distribution and prior parameters of stdev multipliers
   * vars, which are outcomes of several types
     * initial guess
     * fit
     * truth
     * samples, where there are more than one.
 * Measurement data

"""
from math import nan

import numpy as np
import pandas as pd

from cascade.dismod.constants import DensityEnum
from cascade.model.priors import prior_distribution


class _PriorView:
    """Slices to access priors with Distribution objects."""
    def __init__(self, kind, ages, times):
        self._kind = kind
        self.ages = ages
        self.times = times
        self.param_names = ["density", "mean", "std", "lower", "upper", "eta", "nu"]
        # The grid is the age-time grid.
        self.grid = pd.DataFrame(dict(
            prior_name=None,
            kind=kind,
            age=np.repeat(ages, len(times)),
            time=np.tile(times, len(ages)),
            density=None,
            mean=nan,
            lower=nan,
            upper=nan,
            std=nan,
            nu=nan,
            eta=nan,
        ))
        # The mulstd is a dataframe with one entry.
        self.mulstd = pd.DataFrame(dict(
            prior_name=None,
            kind=kind,
            age=[nan],
            time=[nan],
            density=[None],
            mean=[nan],
            lower=[nan],
            upper=[nan],
            std=[nan],
            nu=[nan],
            eta=[nan],
        ))

    def __len__(self):
        if self.mulstd.iloc[0].density is None:
            len_std = 0
        else:
            len_std = 1
        return len(self.grid) + len_std

    def __setitem__(self, at_slice, value):
        """
        Args:
            at_slice (slice, slice): What to change, as integer offset into ages and times.
            value (priors.Prior): The prior to set, containing dictionary of
                                  parameters.
        """
        ages = self.ages[at_slice[0]]
        times = self.times[at_slice[1]]
        to_set = value.parameters()
        to_set["density_id"] = DensityEnum[to_set["density"]].value
        self.grid.loc[
            np.in1d(self.grid.age, ages) & np.in1d(self.grid.time, times)
            & (self.grid.kind == self._kind),
            self.param_names
        ] = [to_set[setp] if setp in to_set else nan for setp in self.param_names]

    def apply(self, transform):
        these_points = (np.in1d(self._parent.priors.age, self._parent.ages) &
                        np.in1d(self._parent.priors.time, self._parent.times) &
                        (self._parent.priors.kind == self._kind))
        for idx, row in self._parent.priors.loc[these_points, self.param_names + ["age", "time"]].iterrows():
            new_distribution = transform(row.age, row.time, prior_distribution(row))
            to_set = new_distribution.parameters()
            to_set["density_id"] = DensityEnum[to_set["density"]].value
            self._parent.priors.iloc[idx, self.param_names] = [
                to_set[setp] if setp in to_set else nan for setp in self.param_names]


class SmoothGrid:
    def __init__(self, age_time_grid):
        """
        The Smooth Grid is a set of priors on an age-time grid.

        Args:
            age_time_grid (Tuple(set,set)): The supporting grid.
        """
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        self._value = _PriorView("value", self.ages, self.times)
        self._dage = _PriorView("dage", self.ages, self.times)
        self._dtime = _PriorView("dtime", self.ages, self.times)

    def __len__(self):
        return len(self._value) + len(self._dage) + len(self._dtime)

    def __str__(self):
        return f"SmoothGrid({len(self.ages), len(self.times)})"

    @property
    def age_time(self):
        return self.ages, self.times

    @property
    def value(self):
        return self._value

    @property
    def dage(self):
        return self._dage

    @property
    def dtime(self):
        return self._dtime

    @property
    def priors(self):
        """All priors in one dataframe."""
        return pd.concat(
            [self._value.grid, self._value.mulstd, self._dage.grid,
             self._dage.mulstd, self.dtime.grid, self.dtime.mulstd]) \
            .reset_index(drop=True)


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
    smooth_grid = SmoothGrid(var.age_time)
    if strictly_positive:
        smooth_grid.value.grid.loc[:, ["density", "mean", "lower", "upper"]] = [
            "uniform", 1e-2, 1e-9, 5
        ]
    else:
        smooth_grid.value.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -5, 5, 0]
    smooth_grid.dage.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -5, 5, 0]
    smooth_grid.dtime.grid.loc[:, ["density", "lower", "upper", "mean"]] = ["uniform", -5, 5, 0]
    return smooth_grid
