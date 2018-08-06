from math import isclose, isinf

GRID_SNAP_DISTANCE = 1 / 365


class AgeTimeGrid:
    @classmethod
    def uniform(cls, age_start, age_end, age_step, time_start, time_end, time_step):
        ages = list(range(age_start, age_end, age_step))
        times = list(range(time_start, time_end, time_step))

        return cls(ages, times)

    def __init__(self, ages, times, snap_distance=GRID_SNAP_DISTANCE):
        self.ages = ages
        self.times = times
        self.snap_distance = snap_distance


def _any_close(value, targets, tolerance):
    for target in targets:
        if isclose(value, target, abs_tol=tolerance):
            return True
    return False


def _validate_region(grid, lower_age, upper_age, lower_time, upper_time):
    if not _any_close(lower_age, grid.ages, grid.snap_distance) and not isinf(lower_age):
        raise ValueError("Lower age not in underlying grid")

    if not _any_close(upper_age, grid.ages, grid.snap_distance) and not isinf(upper_age):
        raise ValueError("Upper age not in underlying grid")

    if not _any_close(lower_time, grid.times, grid.snap_distance) and not isinf(lower_time):
        raise ValueError("Lower time not in underlying grid")

    if not _any_close(upper_time, grid.times, grid.snap_distance) and not isinf(upper_time):
        raise ValueError("Upper time not in underlying grid")


class _RegionView:
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

        return self._parent.prior_at_point(lower_age, lower_time)

    @prior.setter
    def prior(self, p):
        self._parent.push_prior(*self._rectangle(), p)


class Priors:
    def __init__(self, grid):
        self._grid = grid
        self._priors = list()

    def __getitem__(self, slices):
        if len(slices) != 2:
            raise ValueError("Region must be specified in both age and time")

        age_slice, time_slice = slices
        if not isinstance(age_slice, slice):
            age_slice = slice(age_slice, age_slice)
        if not isinstance(time_slice, slice):
            time_slice = slice(time_slice, time_slice)
        if age_slice.step is not None or time_slice.step is not None:
            raise ValueError("Step size must not be specified. It is defined in the underlying grid")

        return _RegionView(self, age_slice, time_slice)

    def push_prior(self, lower_age, upper_age, lower_time, upper_time, prior):
        _validate_region(self._grid, lower_age, upper_age, lower_time, upper_time)
        self._priors.append(((lower_age, upper_age, lower_time, upper_time), prior))

    def prior_at_point(self, age, time):
        final_prior = None
        for ((lower_age, upper_age, lower_time, upper_time), prior) in self._priors:
            if lower_age <= age <= upper_age and lower_time <= time <= upper_time:
                final_prior = prior
        return final_prior
