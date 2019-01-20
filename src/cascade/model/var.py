from itertools import product
from math import nan

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d


class Var:
    """A Var is a set of values of a random field on a SmoothGrid."""
    def __init__(self, age_time_grid, count=1):
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        self.grid = pd.DataFrame(dict(
            age=np.tile(age_time[:, 0], count),
            time=np.tile(age_time[:, 1], count),
            mean=nan,
            idx=np.repeat(range(count), len(age_time)),
        ))
        self.mulstd = dict()  # keys are value, dage, dtime.
        self._spline = None

    def check(self, name=None):
        if not self.grid["mean"].notna().all():
            raise RuntimeError(
                f"Var {name} has {self.grid['mean'].isna().sum()} nan values")
        if set(self.mulstd.keys()) - {"value", "dage", "dtime"}:
            raise RuntimeError(
                f"Var {name} has mulstds besides the three: {list(self.mulstd.keys())}"
            )

    def age_time(self):
        yield from zip(np.repeat(self.ages, len(self.times)), np.tile(self.times, len(self.ages)))

    def __getitem__(self, age_time):
        age, time = age_time
        rows = self.grid.query("age == @age and time == @time")
        if len(rows) > 0:
            return rows["mean"]
        else:
            raise KeyError(f"Age {age} and time {time} not found.")

    def __len__(self):
        return self.ages.shape[0] * self.times.shape[0] + len(self.mulstd)

    def __str__(self):
        return f"Var({len(self.ages), len(self.times)})"

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

    def __call__(self, age, time):
        if self._spline is None:
            self._spline = self._as_function()
        result = self._spline(age, time)
        # Result can be a numpy array, so undo that if input wasn't an array.
        if np.isscalar(age) and np.isscalar(time):
            return result.item()
        else:
            return result

    def _as_function(self):
        """Constructs a function which mimics how Dismod-AT turns a field of
        points in age and time into a continuous function.

        Returns:
            function: Of age and time.
        """
        age_time_df = self.grid
        ordered = age_time_df.sort_values(["age", "time"])
        age = np.sort(np.unique(age_time_df.age.values))
        time = np.sort(np.unique(age_time_df.time.values))
        if len(age) > 1 and len(time) > 1:
            spline = RectBivariateSpline(age, time, ordered["mean"].values.reshape(len(age), len(time)), kx=1, ky=1)

            def bivariate_function(x, y):
                return spline(x, y)[0]

            return bivariate_function

        elif len(age) * len(time) > 1:
            fill = (ordered["mean"].values[0], ordered["mean"].values[-1])
            independent = age if len(age) != 1 else time
            spline = interp1d(
                independent, ordered["mean"].values, kind="linear", bounds_error="extrapolate", fill_value=fill)

            def age_spline(x, _):
                return spline(x)

            def time_spline(_, y):
                return spline(y)

            if len(age) != 1:
                return age_spline
            else:
                return time_spline
        elif len(age) == 1 and len(time) == 1:

            def constant_everywhere(_a, _t):
                return ordered["mean"].values[0]

            return constant_everywhere
        else:
            raise RuntimeError(f"Cannot interpolate if ages or times are length zero: "
                               f"ages {len(age)} times {len(time)}")
