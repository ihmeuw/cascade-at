import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from cascade.model.age_time_grid import AgeTimeGrid


class Var(AgeTimeGrid):
    """A Var is a set of values of a random field on a SmoothGrid.

    It's an AgeTimeGrid, but AgeTimeGrids present each data item
    as a row. This presents each data item as a float. It also
    behaves like a continuous function of age and time.
    """
    def __init__(self, age_time_grid):
        super().__init__(age_time_grid, columns=["mean"])
        self._spline = None

    def check(self, name=None):
        """None of the means should be nan. There should only be the
        three mulstds."""
        if not self.grid["mean"].notna().all():
            raise RuntimeError(
                f"Var {name} has {self.grid['mean'].isna().sum()} nan values")
        if set(self.mulstd.keys()) - {"value", "dage", "dtime"}:
            raise RuntimeError(
                f"Var {name} has mulstds besides the three: {list(self.mulstd.keys())}"
            )

    def __setitem__(self, at_slice, value):
        """
        Args:
            at_slice (slice, slice): What to change, as integer offset into
                ages and times.
            value (float): Set with a single floating-point value.
        """
        super().__setitem__(at_slice, [value])

    def __getitem__(self, item):
        return float(super().__getitem__(item)["mean"])

    def __str__(self):
        return f"Var({len(self.ages), len(self.times)})"

    def __call__(self, age, time):
        """Call a Var as a function of age and time.

        The grid points in a Var represent a continuous function, determined
        by bivariate interpolation. All points outside the grid are equal
        to the nearest point inside the grid.
        """
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
                independent, ordered["mean"].values, kind="linear", bounds_error=False, fill_value=fill)

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
