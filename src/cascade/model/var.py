import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from cascade.dismod.constants import PriorKindEnum
from cascade.model.age_time_grid import AgeTimeGrid


class Var(AgeTimeGrid):
    """A Var is a function of age and time, defined by values on a grid.
    It linearly interpolates over values defined at grid points in a
    rectangular grid of age and time.

    This is a single age-time grid. It is usually found in
    :py:class:`cascade.model.DismodGroups`
    object which is a set of age-time grids. The following are
    ``DismodGroups`` containing
    :py:class:`cascade.model.Var`: the fit, initial guess, truth var,
    and scale var.

    Args:
        ages (List[float]): Points along the age axis.
        times (List[float]): Points in time.
        column_name (str): A var has an internal Pandas DataFrame
            representation, and this column name can be ``mean``
            or ``meas_value``, depending on which Var is needed.
    """
    def __init__(self, ages, times, column_name="mean"):
        self._column_name = column_name
        super().__init__(ages, times, columns=self._column_name)
        self._spline = None

    def check(self, name=None):
        """This raises a :py:class:`ValueError` if any part of the
        Var is uninitialized. None of the means should be nan. There should only be the
        three mulstds."""
        if not self.grid[self._column_name].notna().all():
            raise ValueError(
                f"Var {name} has {self.grid[self._column_name].isna().sum()} nan values")
        if set(self.mulstd.keys()) - {"value", "dage", "dtime"}:
            raise ValueError(
                f"Var {name} has mulstds besides the three: {list(self.mulstd.keys())}"
            )

    def __setitem__(self, at_slice, value):
        """
        To set a value on a Var instance, set it on ranges of age and
        time or at specific ages and times.

        >>> var = Var([0, 10, 20], [2000])
        >>> var[:, :] = 0.001
        >>> var[5:50, 2000] = 0.01
        >>> var[10, :] = 0.02

        Args:
            at_slice (slice, slice): What to change, as integer offset into
                ages and times.
            value (float): A float or integer.
        """
        super().__setitem__(at_slice, [value])

    def __getitem__(self, age_and_time):
        """
        Gets the value of a Var at a single point. The point has to be
        one of the ages and times defined when the var was created.

        >>> var = Var([0, 50, 100], [1990, 2000, 2010])
        >>> var[:, :] = 1e-4
        >>> assert var[50, 2000] == 1e-4

        Trying to read from an age and time not in the ages and times
        of the grid will result in a :py:class:`KeyError`.

        An easy way to set values is to use the `age_time` iterator,
        which loops through the ages and times in the underlying grid.

        >>> for age, time in var.age_time():
        >>>    var[age, time] = 0.01 * age

        Args:
            age_and_time (age, time): A two-dimensional index of age and time.

        Returns:
            float: The value at this age and time.
        """
        return float(super().__getitem__(age_and_time)[self._column_name])

    def set_mulstd(self, kind, value):
        """Set the value of the multiplier on the standard deviation.
        Kind must be one of
        "value", "dage", or "dtime". The value should be convertible to a float.

        >>> var = Var([50], [2000, 2001, 2002])
        >>> var.set_mulstd("value", 0.4)

        """
        sig = "kind is one of value, dage, dtime, and value is a float."
        if kind not in PriorKindEnum.__members__:
            raise ValueError(f"{sig} kind={kind}")
        self.mulstd[kind].loc[:, self._column_name] = float(value)

    def get_mulstd(self, kind):
        """
        Get the value of a standard deviation multiplier for a Var.

        >>> var = Var([50], [2000, 2001, 2002])
        >>> var.set_mulstd("value", 0.4)
        >>> assert var.get_mulstd("value") == 4

        If the standard deviation multiplier wasn't set, then this will
        return a nan.

        >>> assert np.isnan(var.get_mulstd("dage"))

        """
        if kind not in PriorKindEnum.__members__:
            raise ValueError(f"Argument is one of value, dage, dtime, not {kind}.")
        return float(self.mulstd[kind][self._column_name])

    def __str__(self):
        return f"Var({len(self.ages), len(self.times)})"

    def __call__(self, age, time):
        """A Var is a function of age and time, and this is how to call it.

        >>> var = Var([0, 100], [1990, 2000])
        >>> var[0, 1990] = 0
        >>> var[0, 2000] = 1
        >>> var[100, 1990] = 2
        >>> var[100, 2000] = 3
        >>> for a, t in var.age_time():
        >>>     print(f"At corner ({a}, {t}), {var(a, t)}")
        >>> for a, ti in [[53, 1997], [-5, 2000], [120, 2000], [0, 1900], [0, 2010]]:
        >>>     print(f"Anywhere ({a}, {t}), {var(a, t)}")
        At corner (0.0, 1990.0), 0.0
        At corner (0.0, 2000.0), 1.0
        At corner (100.0, 1990.0), 2.0
        At corner (100.0, 2000.0), 3.0
        Anywhere (53, 2000.0), 2.06
        Anywhere (-5, 2000.0), 1.0
        Anywhere (120, 2000.0), 3.0
        Anywhere (0, 2000.0), 1.0
        Anywhere (0, 2000.0), 1.0

        The grid points in a Var represent a continuous function, determined
        by bivariate interpolation. All points outside the grid are equal
        to the nearest point inside the grid.
        """
        if self._spline is None:
            self._spline = self._as_function()
        result = self._spline(age, time)
        # Result can be a numpy array, so undo that if input wasn't an array.
        if np.isscalar(age) and np.isscalar(time):
            return result.item()  # Numpy array has item().
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
            heights = ordered[self._column_name].values.reshape(len(age), len(time))
            spline = RectBivariateSpline(age, time, heights, kx=1, ky=1)

            def bivariate_function(x, y):
                return spline(x, y)[0]

            return bivariate_function

        elif len(age) * len(time) > 1:
            fill = (ordered[self._column_name].values[0], ordered[self._column_name].values[-1])
            independent = age if len(age) != 1 else time
            spline = interp1d(
                independent, ordered[self._column_name].values, kind="linear", bounds_error=False, fill_value=fill)

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
                return ordered[self._column_name].values[0]

            return constant_everywhere
        else:
            raise RuntimeError(f"Cannot interpolate if ages or times are length zero: "
                               f"ages {len(age)} times {len(time)}")
