from datetime import timedelta
from itertools import product
from math import nan, inf

import numpy as np
import pandas as pd

from cascade.dismod.constants import PriorKindEnum

GRID_SNAP_DISTANCE = 1 / timedelta(days=365).total_seconds()
"""Times within one second are considered equal."""


class AgeTimeGrid:
    """A Var is a set of values of a random field on a SmoothGrid.
    At each age and time point is a dataframe consisting of the columns
    given in the constructor. So getting an item returns a dataframe
    with those columns. Setting a dataframe sets those columns.
    Each AgeTimeGrid has three possible mulstds, for value, dage, dtime.
    """
    def __init__(self, age_time_grid, columns, count=1):
        assert isinstance(columns[0], str)
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        self.columns = columns
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        self.grid = pd.DataFrame(dict(
            age=np.tile(age_time[:, 0], count),
            time=np.tile(age_time[:, 1], count),
            idx=np.repeat(range(count), len(age_time)),
        ))
        self.grid = self.grid.assign(**{new_col: nan for new_col in columns})
        self.mulstd = dict()
        # Each mulstd is one line.
        for kind in PriorKindEnum:
            mulstd_df = pd.DataFrame(dict(
                age=[nan],
                time=[nan],
                idx=list(range(count)),
            ))
            mulstd_df = mulstd_df.assign(**{new_col: nan for new_col in columns})
            self.mulstd[kind.name] = mulstd_df

    def age_time(self):
        yield from zip(np.repeat(self.ages, len(self.times)), np.tile(self.times, len(self.ages)))

    def __getitem__(self, age_time):
        age, time = age_time
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
        mulstd_cnt = sum(not df[self.columns].dropna(how="all").empty for df in self.mulstd.values())
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
