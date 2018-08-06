from math import isclose, isinf

GRID_SNAP_DISTANCE = 1 / 365


class IdFactory:
    def __init__(self):
        self._objects = {}

    def __call__(self, to_id):
        object_hash = hash(to_id)
        if object_hash not in self._objects:
            if not self._objects:
                new_id = 0
            else:
                new_id = max(self._objects.values()) + 1

            self._objects[object_hash] = new_id

        return self._objects[object_hash]


class Prior:
    def __init__(self, name=None):
        self.name = name

    def _parameters(self):
        raise NotImplementedError()

    def _full_parameters(self):
        return dict(prior_name=self.name, **self._parameters())

    def __hash__(self):
        return hash(frozenset(self._full_parameters().items()))

    def __eq__(self, other):
        return isinstance(other, Prior) and self._full_parameters() == other._full_parameters()

    def db_row(self, id_factory):
        row = {
            "prior_name": None,
            "density": None,
            "lower": None,
            "upper": None,
            "mean": None,
            "std": None,
            "eta": None,
            "nu": None,
        }.update(self._parameters())

        prior_id = id_factory(self)

        if row["prior_name"] is None:
            row["prior_name"] = f"prior_{prior_id}"

        return dict(prior_id=prior_id, **row)


class UniformPrior(Prior):
    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value

    def _parameters(self):
        return {"density": "uniform", "lower": self.value, "upper": self.value, "mean": self.value}


class GaussianPrior(Prior):
    def __init__(self, mean, standard_deviation, lower=float("-inf"), upper=float("inf"), name=None):
        super().__init__(name=name)
        if not lower <= mean <= upper:
            raise ValueError("Bounds are inconsistent")
        if standard_deviation < 0:
            raise ValueError("Standard deviation must be positive")

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation

    def _parameters(self):
        return {
            "density": "gaussian",
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
        }


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
