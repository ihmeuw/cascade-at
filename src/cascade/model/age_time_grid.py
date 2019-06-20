from datetime import timedelta
from itertools import product
from math import nan, inf

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.constants import PriorKindEnum

CODELOG, MATHLOG = getLoggers(__name__)
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

    Why is this in Pandas, when it's a regular array of data with an
    index, which makes it better suited to XArray, or even a
    Numpy array? It needs to interface with a database representation,
    and Pandas is a better match there.
    """
    def __init__(self, ages, times, columns):
        try:
            self.ages = np.sort(np.atleast_1d(ages).astype(np.float))
            self.times = np.sort(np.atleast_1d(times).astype(np.float))
        except TypeError:
            raise TypeError(f"Ages and times should be arrays of floats {(ages, times)}.")
        type_constraint = "Columns should be either a string or an iterable of strings."
        if isinstance(columns, str):
            columns = [columns]
        try:
            self.columns = list(columns)
        except TypeError:
            raise TypeError(f"{type_constraint} {columns}")
        for col_is_str in self.columns:
            if not isinstance(col_is_str, str):
                raise TypeError(f"{type_constraint} {col_is_str}")
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        self.grid = pd.DataFrame(dict(
            age=age_time[:, 0],
            time=age_time[:, 1],
        ))
        self.grid = self.grid.assign(**{new_col: nan for new_col in columns})
        self._mulstd = dict()
        # Each mulstd is one record.
        for kind in PriorKindEnum:
            mulstd_df = pd.DataFrame(dict(
                age=[nan],
                time=[nan],
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
            raise ValueError(f"No ages within range {at_range[0]} "
                             "Are you looking for a point not in the grid?")
        if len(times) == 0:
            raise ValueError(f"No times within range {at_range[1]} "
                             "Are you looking for a point not in the grid?")
        self.grid.loc[np.in1d(self.grid.age, ages) & np.in1d(self.grid.time, times), self.columns] = value

    def __len__(self):
        return self.variable_count()

    def variable_count(self):
        mulstd_cnt = sum(not df[self.columns].dropna(how="all").empty for df in self._mulstd.values())
        return self.ages.shape[0] * self.times.shape[0] + mulstd_cnt

    def __str__(self):
        return f"AgeTimeGrid({len(self.ages)}, {len(self.times)}) with {self.variable_count()} model variables."

    def __repr__(self):
        return f"AgeTimeGrid({self.ages}, {self.times})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            CODELOG.debug(f"SmoothGrid not equal to {other}")
            return NotImplemented
        if set(self.mulstd.keys()) != set(other.mulstd.keys()):
            CODELOG.debug(f"Different number of mulstd keys")
            return False
        for mul_key in self.mulstd.keys():
            try:
                pd.testing.assert_frame_equal(self.mulstd[mul_key], other.mulstd[mul_key])
            except AssertionError:
                CODELOG.debug("assert frame equal false on mulstd")
                return False
        try:
            pd.testing.assert_frame_equal(self.grid, other.grid, check_like=True, check_exact=False)
            return True
        except AssertionError as ae:
            if "values are different" in str(ae):
                CODELOG.debug("assert frame equal false on grid")
                return False
            else:
                raise
