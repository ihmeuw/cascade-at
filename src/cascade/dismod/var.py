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

    @property
    def age_time(self):
        return self.ages, self.times

    def __len__(self):
        return self.ages.shape[0] * self.times.shape[0] + len(self.mulstd)

    def __str__(self):
        return f"Var({len(self.ages), len(self.times)})"

    def as_function(self):
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

        elif len(age) == 1 or len(time) == 1:
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
        else:
            raise RuntimeError(f"Cannot interpolate if ages or times are length zero: "
                               f"ages {len(age)} times {len(time)}")
