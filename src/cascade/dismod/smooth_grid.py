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
from itertools import product
from math import nan

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import DensityEnum
from cascade.model.priors import Uniform

PRIOR_KINDS = ["value", "dage", "dtime"]


class _PriorView:
    """Slices to access priors with Distribution objects."""
    def __init__(self, parent, kind):
        self._kind = kind
        self._parent = parent
        self.param_names = ["density_id", "mean", "std", "lower", "upper", "eta", "nu"]

    def __setitem__(self, at_slice, value):
        """
        Args:
            at_slice (slice, slice): What to change, as integer offset into ages and times.
            value (priors.Prior): The prior to set, containing dictionary of
                                  parameters.
        """
        ages = self._parent.ages[at_slice[0]]
        times = self._parent.times[at_slice[1]]
        to_set = value.parameters()
        to_set["density_id"] = DensityEnum[to_set["density"]].value
        self._parent.priors.loc[
            np.in1d(self._parent.priors.age, ages) & np.in1d(self._parent.priors.time, times)
            & (self._parent.priors.kind == self._kind),
            self.param_names
        ] = [to_set[setp] if setp in to_set else nan for setp in self.param_names]


class SmoothGrid:
    def __init__(self, age_time_grid):
        """
        The Smooth Grid is a set of priors on an age-time grid.

        Args:
            age_time_grid (Tuple(set,set)): The supporting grid.
        """
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        self.parameter_columns = ["density_id", "mean", "lower", "upper", "std", "nu", "eta"]
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        one_priors = pd.DataFrame(dict(
            prior_name=None,
            kind="value",
            age=age_time[:, 0],
            time=age_time[:, 1],
            density_id=nan,
            mean=nan,
            lower=nan,
            upper=nan,
            std=nan,
            nu=nan,
            eta=nan,
        ))
        one_priors = one_priors.append({"age": nan, "time": nan, "kind": "value", "name": None}, ignore_index=True)
        self.priors = pd.concat([one_priors, one_priors.assign(kind="dage"), one_priors.assign(kind="dtime")])
        self.priors = self.priors.reset_index(drop=True)

    def __len__(self):
        mulstd = len(self.priors[self.priors.age.isna() & self.priors.density_id.notna()])
        return self.ages.shape[0] * self.times.shape[0] * 3 + mulstd

    def __str__(self):
        return f"SmoothGrid({len(self.ages), len(self.times)})"

    @property
    def age_time(self):
        return (self.ages, self.times)

    @property
    def value(self):
        return _PriorView(self, "value")

    @property
    def dage(self):
        return _PriorView(self, "dage")

    @property
    def dtime(self):
        return _PriorView(self, "dtime")


def smooth_grid_from_var(var):
    """

    Args:
        var (Var): A single var grid.

    Returns:
        SmoothGrid: A single smooth grid with Uniform distributions.
    """
    smooth_grid = SmoothGrid(var.age_time)
    smooth_grid[:, :] = Uniform(-5, 5, 0)
    smooth_grid.priors["mean"] = var.grid["mean"]
    return smooth_grid
