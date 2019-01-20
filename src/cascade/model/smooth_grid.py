from datetime import timedelta
from math import nan, inf
from numbers import Real

import numpy as np
import pandas as pd

from cascade.model.priors import prior_distribution

GRID_SNAP_DISTANCE = 1 / timedelta(days=365).total_seconds()
"""Times within one second are considered equal."""


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

    def age_time(self):
        yield from zip(np.repeat(self.ages, len(self.times)), np.tile(self.times, len(self.ages)))

    def age_time_diff(self):
        yield from zip(
            np.repeat(self.ages, len(self.times)),
            np.tile(self.times, len(self.ages)),
            np.repeat(np.ediff1d(self.ages, to_end=inf), len(self.times)),
            np.tile(np.ediff1d(self.times, to_end=inf), len(self.ages)),
        )

    def __len__(self):
        if self.mulstd.iloc[0].density is None:
            len_std = 0
        else:
            len_std = 1
        return len(self.grid) + len_std

    def __getitem__(self, at_slice):
        try:
            if len(at_slice) != 2:
                raise ValueError("Set value at an age and time, so two arguments.")
            for check_int in at_slice:
                if not isinstance(check_int, Real):
                    raise ValueError(f"Can only get one prior so use age-time values, not slices.")
        except TypeError:
            raise ValueError("Set value at an age and time, so two arguments")
        age_time = (self.ages, self.times)
        at_idx = list()
        for idx in range(2):
            closest = age_time[idx][np.abs(age_time[idx] - at_slice[idx]).argmin()]
            if not np.isclose(closest, at_slice[idx], atol=GRID_SNAP_DISTANCE):
                raise ValueError(f"The nearest point to {at_slice[idx]} is {closest}.")
            at_idx.append(closest)
        return prior_distribution(self.grid[(self.grid.age == at_idx[0]) & (self.grid.time == at_idx[1])].iloc[0])

    def __setitem__(self, at_slice, value):
        """
        Args:
            at_slice (slice, slice): What to change, as integer offset into ages and times.
            value (priors.Prior): The prior to set, containing dictionary of
                                  parameters.
        """
        try:
            if len(at_slice) != 2:
                raise ValueError("Set value at an age and time, so two arguments.")
        except TypeError:
            raise ValueError("Set value at an age and time, so two arguments")
        at_range = list()
        for one_slice in at_slice:
            if not isinstance(one_slice, slice):
                one_slice = slice(one_slice, one_slice)
            if one_slice.step is not None:
                raise ValueError("Slice in age or time, without a step.")
            start = one_slice.start if one_slice.start is not None else -inf
            stop = one_slice.stop if one_slice.stop is not None else inf
            at_range.append([start - GRID_SNAP_DISTANCE, stop + GRID_SNAP_DISTANCE])
        ages = self.ages[(at_range[0][0] <= self.ages) & (self.ages <= at_range[0][1])]
        times = self.times[(at_range[1][0] <= self.times) & (self.times <= at_range[1][1])]
        if len(ages) == 0:
            raise ValueError(f"No ages within range {at_range[0]}")
        if len(times) == 0:
            raise ValueError(f"No times within range {at_range[1]}")
        to_set = value.parameters()
        self.grid.loc[
            np.in1d(self.grid.age, ages) & np.in1d(self.grid.time, times)
            & (self.grid.kind == self._kind),
            self.param_names
        ] = [to_set[setp] if setp in to_set else nan for setp in self.param_names]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.mulstd != other.mulstd:
            return False
        try:
            pd.testing.assert_frame_equal(self.grid, other.grid)
            pd.testing.assert_frame_equal(self.mulstd, other.mulstd)
            return True
        except AssertionError as ae:
            if "values are different" in str(ae):
                return False
            else:
                raise

    def apply(self, transform):
        for idx, row in self.grid.loc[:, self.param_names + ["age", "time"]].iterrows():
            new_distribution = transform(row.age, row.time, prior_distribution(row))
            to_set = new_distribution.parameters()
            self.grid.iloc[idx, self.param_names] = [
                to_set[setp] if setp in to_set else nan for setp in self.param_names]


class SmoothGrid:
    def __init__(self, age_time_grid):
        """
        The Smooth Grid is a set of priors on an age-time grid.

        Args:
            age_time_grid (Tuple(set,set)): The supporting grid.
        """
        self.ages = np.sort(np.array(age_time_grid[0], dtype=np.float))
        self.times = np.sort(np.array(age_time_grid[1], dtype=np.float))
        self._value = _PriorView("value", self.ages, self.times)
        self._dage = _PriorView("dage", self.ages, self.times)
        self._dtime = _PriorView("dtime", self.ages, self.times)

    def __len__(self):
        return len(self._value) + len(self._dage) + len(self._dtime)

    def __str__(self):
        return f"SmoothGrid({len(self.ages), len(self.times)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._value == other._value and self._dage == other._dage and self._dtime == other._dtime

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
