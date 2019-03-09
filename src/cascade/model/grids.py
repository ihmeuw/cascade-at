"""Support for dismod concepts that are expressed over grids: age-time, smoothings and value priors.
"""

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
    def uniform(cls, age_lower, age_upper, age_step, time_lower, time_upper, time_step):
        """Construct an age-time grid with uniform spacing on each axis.
        All values are in units of years.

        Args:
            age_lower: the lowest age in the grid
            age_upper: the highest age in the grid (inclusive)
            age_step: the step between ages
            time_lower: the earliest time in the grid
            time_upper: the latest time in the grid (inclusive)
            time_step: the step between times
        """
        ages = np.arange(age_lower, age_upper, age_step)
        times = np.arange(time_lower, time_upper, time_step)

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
