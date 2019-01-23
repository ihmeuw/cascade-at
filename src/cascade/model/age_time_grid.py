from datetime import timedelta
from itertools import product
from math import nan, inf

import numpy as np
import pandas as pd

from cascade.dismod.constants import PriorKindEnum

GRID_SNAP_DISTANCE = 1 / timedelta(days=365).total_seconds()
"""Times within one second are considered equal."""


class AgeTimeGrid:
    """The AgeTime grid holds rows of a table at each age and time value.

    At each age and time point is a DataFrame consisting of the columns
    given in the constructor. So getting an item returns a dataframe
    with those columns. Setting a DataFrame sets those columns.
    Each AgeTimeGrid has three possible mulstds, for value, dage, dtime.

    >>> atg = AgeTimeGrid([0, 10, 20], [1990, 2000, 2010], ["height", "weight"])
    >>> atg[:, :] = [6.1, 195]
    >>> atg[:, :].height = [5.9]
    >>> atg[10, 2000] = [5.7, 180]
    >>> atg[5:17, 1980:1990].weight = 125
    >>> assert (atg[20, 2000].weight == 195).all()
    >>> assert isinstance(atg[0, 1990], pd.DataFrame)

    If the column has the same name as a function (mean), then access it
    with getitem,

    >>> atg[:, :]["mean"] = [5.9]

    """
    def __init__(self, ages, times, columns, count=1):
        assert isinstance(columns[0], str)
        try:
            self.ages = np.array(ages, dtype=np.float)
            self.times = np.array(times, dtype=np.float)
        except TypeError:
            raise TypeError(f"Ages and times should be arrays of floats {(ages, times)}.")
        if isinstance(columns, str):
            columns = [columns]
        try:
            self.columns = list(columns)
        except TypeError:
            raise TypeError(f"Columns should be an iterable of strings. {columns}")
        for col_is_str in self.columns:
            if not isinstance(col_is_str, str):
                raise TypeError(f"Columns should be iterable of strings. {col_is_str}")
        try:
            count = int(count)
        except ValueError:
            raise TypeError(f"Count must be an integer {count}.")

        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        self.grid = pd.DataFrame(dict(
            age=np.tile(age_time[:, 0], count),
            time=np.tile(age_time[:, 1], count),
            idx=np.repeat(range(count), len(age_time)),
        ))
        self.grid = self.grid.assign(**{new_col: nan for new_col in columns})
        self._mulstd = dict()
        # Each mulstd is one record.
        for kind in PriorKindEnum:
            mulstd_df = pd.DataFrame(dict(
                age=[nan],
                time=[nan],
                idx=list(range(count)),
            ))
            mulstd_df = mulstd_df.assign(**{new_col: nan for new_col in columns})
            self._mulstd[kind.name] = mulstd_df

    @property
    def mulstd(self):
        return self._mulstd

    def age_time(self):
        yield from zip(np.repeat(self.ages, len(self.times)), np.tile(self.times, len(self.ages)))

    def __getitem__(self, age_time):
        """
        Args:
            age_time (float, float): Gets all rows with this (age, time).

        Returns:
            pd.DataFrame or pd.Series with columns.
        """
        try:
            age, time = age_time
        except TypeError as te:
            if "not iterable" in str(te):
                raise TypeError(f"Index should be two floats for getting, not {age_time}.")
            else:
                raise
        if isinstance(age, slice) or isinstance(time, slice):
            raise TypeError(f"Cannot get a slice from an AgeTimeGrid.")
        rows = self.grid.query("age == @age and time == @time")
        if len(rows) > 0:
            return rows[self.columns]
        else:
            raise KeyError(f"Age {age} and time {time} not found.")

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
        self.grid.loc[np.in1d(self.grid.age, ages) & np.in1d(self.grid.time, times), self.columns] = value

    def __len__(self):
        mulstd_cnt = sum(not df[self.columns].dropna(how="all").empty for df in self._mulstd.values())
        return self.ages.shape[0] * self.times.shape[0] + mulstd_cnt

    def __str__(self):
        return f"AgeTimeGrid({len(self.ages), len(self.times)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.mulstd != other.mulstd:
            return False
        try:
            pd.testing.assert_frame_equal(self.grid, other.grid)
            return True
        except AssertionError as ae:
            if "values are different" in str(ae):
                return False
            else:
                raise
