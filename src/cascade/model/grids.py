"""Support for dismod concepts that are expressed over grids: age-time, smoothings and value priors.
"""
from math import isclose, isinf

import numpy as np

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

GRID_SNAP_DISTANCE = 2e-16


def _nearest(values, query):
    values = np.array(values)
    return values[np.abs(values - query).argmin()]


def unique_floats(values):
    values = np.array(values)
    uniqued_values = np.unique(values.round(decimals=14), return_index=True)[1]

    return values[uniqued_values]


class AgeTimeGrid:
    """Structure for storing age and times over which the model will estimate.
    Also constructors for making standard age-time grids.
    """

    @classmethod
    def uniform(cls, age_start, age_end, age_step, time_start, time_end, time_step):
        """Construct an age-time grid with uniform spacing on each axis.
        All values are in units of years.

        Args:
            age_start: the lowest age in the grid
            age_end: the highest age in the grid (inclusive)
            age_step: the step between ages
            time_start: the earliest time in the grid
            time_end: the latest time in the grid (inclusive)
            time_step: the step between times
        """
        ages = np.arange(age_start, age_end, age_step)
        times = np.arange(time_start, time_end, time_step)

        return cls(ages, times)

    def __init__(self, ages, times):
        self._ages = tuple(sorted(unique_floats(ages)))
        self._times = tuple(sorted(unique_floats(times)))

    @property
    def ages(self):
        return self._ages

    @property
    def times(self):
        return self._times

    def to_age(self, query):
        if np.isinf(query):
            return query
        return _nearest(self._ages, query)

    def to_time(self, query):
        if np.isinf(query):
            return query
        return _nearest(self._times, query)

    def __hash__(self):
        return hash((self._ages, self._times))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._ages == other._ages and self._times == other._times


def _any_close(value, targets):
    """True if any element of targets is within tolerance of value
    """
    for target in targets:
        if isclose(value, target, abs_tol=GRID_SNAP_DISTANCE):
            return True
    return False


def _validate_region(grid, lower_age, upper_age, lower_time, upper_time):
    """Validates that the described rectangle aligns with the grid.
    Positive or negative infinite bounds will be treated as stretching to the edge of the grid.

    Raises:
        ValueError: If the rectangle does not align
    """
    if not _any_close(lower_age, grid.ages) and not isinf(lower_age):
        raise ValueError("Lower age not in underlying grid")

    if not _any_close(upper_age, grid.ages) and not isinf(upper_age):
        raise ValueError("Upper age not in underlying grid")

    if not _any_close(lower_time, grid.times) and not isinf(lower_time):
        raise ValueError("Lower time not in underlying grid")

    if not _any_close(upper_time, grid.times) and not isinf(upper_time):
        raise ValueError("Upper time not in underlying grid")


class _RegionView:
    """Represents a slice into a PriorGrid and exposes a way to query or set
    the prior over that slice.

    Args:
        parent: The PriorGrid this _RegionView references
        age_slice: The region in age space
        time_slice: The region in time space
    """

    def __init__(self, parent, age_slice, time_slice):
        self._parent = parent
        self._age_slice = age_slice
        self._time_slice = time_slice

    def _rectangle(self):
        if self._age_slice.start is None:
            lower_age = float("-inf")
        else:
            lower_age = self._age_slice.start

        if self._age_slice.stop is None:
            upper_age = float("inf")
        else:
            upper_age = self._age_slice.stop

        if self._time_slice.start is None:
            lower_time = float("-inf")
        else:
            lower_time = self._time_slice.start

        if self._time_slice.stop is None:
            upper_time = float("inf")
        else:
            upper_time = self._time_slice.stop

        return (lower_age, upper_age, lower_time, upper_time)

    @property
    def prior(self):
        lower_age, upper_age, lower_time, upper_time = self._rectangle()
        if lower_age != upper_age or lower_time != upper_time:
            raise NotImplementedError("Currently only point queries are supported")

        point = self._parent._prior_at_point(lower_age, lower_time)
        return point

    @prior.setter
    def prior(self, p):
        self._parent._push_prior(*self._rectangle(), p)


class PriorGrid:
    """Represents priors for rectangular regions of an underlying age-time grid.

    Args:
        grid: The underlying age-time grid
        hyper_prior: The prior for the priors in this grid

    Examples:
        >>> grid = AgeTimeGrid.uniform(age_start=0,age_end=120,age_step=5,time_start=1990,time_end=2018,time_step=1)
        >>> d_time = PriorGrid(grid)
        >>> #Set a prior for the whole grid:
        >>> d_time[:, :].prior = Gaussian(0, 0.1)
        >>> #Set a prior for a band of ages
        >>> d_time[0:15, :].prior = Gaussian(1, 0.01)
        >>> #Or a single year
        >>> d_time[:, 1995].prior = Gaussian(0, 3)
    """

    def __init__(self, grid, hyper_prior=None):
        self.grid = grid
        self._priors = list()
        self.hyper_prior = hyper_prior

    def __getitem__(self, slices):
        try:
            if len(slices) != 2:
                raise ValueError("Region must be specified in both age and time")
        except TypeError:
            raise ValueError("Region must be specified in both age and time")

        age_slice, time_slice = slices
        if not isinstance(age_slice, slice):
            age_slice = slice(age_slice, age_slice)
        if not isinstance(time_slice, slice):
            time_slice = slice(time_slice, time_slice)
        if age_slice.step is not None or time_slice.step is not None:
            raise ValueError("Step size must not be specified. It is defined in the underlying grid")

        return _RegionView(self, age_slice, time_slice)

    def _push_prior(self, lower_age, upper_age, lower_time, upper_time, prior):
        """Push a new region and prior combination onto the stack. When
        querying, more recent priors take precedence.
        """
        _validate_region(self.grid, lower_age, upper_age, lower_time, upper_time)
        lower_age = self.grid.to_age(lower_age)
        upper_age = self.grid.to_age(upper_age)
        lower_time = self.grid.to_time(lower_time)
        upper_time = self.grid.to_time(upper_time)
        self._priors.append(((lower_age, upper_age, lower_time, upper_time), prior))

    def _prior_at_point(self, age, time):
        """Find the prior for a particular point on the age-time grid.
        """
        age = self.grid.to_age(age)
        time = self.grid.to_time(time)
        for ((lower_age, upper_age, lower_time, upper_time), prior) in reversed(self._priors):
            if lower_age <= age <= upper_age and lower_time <= time <= upper_time:
                return prior
        return None

    @property
    def priors(self):
        ps = {prior for _, prior in self._priors}
        if self.hyper_prior:
            ps.add(self.hyper_prior)
        return ps

    def __hash__(self):
        return hash((self.grid, tuple(self._priors), self.hyper_prior))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.grid == other.grid and self._priors == other._priors and self.hyper_prior == other.hyper_prior
