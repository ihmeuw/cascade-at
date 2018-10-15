class Smooth:
    __slots__ = ["_value_priors", "_d_age_priors", "_d_time_priors", "name"]

    def __init__(self, value_priors=None, d_age_priors=None, d_time_priors=None, name=None):
        self._value_priors = None
        self._d_age_priors = None
        self._d_time_priors = None
        self.name = name

        self.value_priors = value_priors
        self.d_age_priors = d_age_priors
        self.d_time_priors = d_time_priors

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.name == other.name
            and self._value_priors == other._value_priors
            and self._d_age_priors == other._d_age_priors
            and self._d_time_priors == other._d_time_priors
        )

    def _validate_grids(self, priors):
        grids = [ps.grid for ps in [self.value_priors, self.d_age_priors, self.d_time_priors, priors] if ps]
        if grids:
            if not all([grids[0] == g for g in grids]):
                raise ValueError("Smooth cannot contain priors on heterogeneous grids")

    def write(self, out_stream, assigned_name=None):
        name = assigned_name if assigned_name else self.name
        my_id = out_stream.append("smooth", [name, n_age, n_time, mul_std])

        # This is the age-time grid
        for grid_point in self.grid:
            age_id = out_stream
            value_id = grid_point.write("prior", value_prior(grid_point))
            age_id = grid_point.append("prior", age_prior(grid_point))
            time_id = grid_point.append("prior", time_prior(grid_point))
            out_stream.append(
                "smooth_grid",
                [my_id, grid_point.age, grid_point.time, value_id, age_id, time_id])



    @property
    def value_priors(self):
        return self._value_priors

    @value_priors.setter
    def value_priors(self, priors):
        self._validate_grids(priors)
        self._value_priors = priors

    @property
    def d_age_priors(self):
        return self._d_age_priors

    @d_age_priors.setter
    def d_age_priors(self, priors):
        self._validate_grids(priors)
        self._d_age_priors = priors

    @property
    def d_time_priors(self):
        return self._d_time_priors

    @d_time_priors.setter
    def d_time_priors(self, priors):
        self._validate_grids(priors)
        self._d_time_priors = priors

    @property
    def grid(self):
        for ps in [self.value_priors, self.d_age_priors, self.d_time_priors]:
            if ps:
                return ps.grid
        return None

    @property
    def prior_grids(self):
        return [ps for ps in [self.value_priors, self.d_age_priors, self.d_time_priors] if ps]


class Rate:
    __slots__ = ["name", "parent_smooth", "child_smoothings", "covariate_multipliers"]

    def __init__(self, name, parent_smooth=None, child_smoothings=None):
        self.name = name
        self.parent_smooth = parent_smooth
        if child_smoothings is None:
            child_smoothings = []
        self.child_smoothings = child_smoothings
        self.covariate_multipliers = []

    @property
    def smoothings(self):
        smoothings = [s for _, s in self.child_smoothings]
        if self.parent_smooth:
            smoothings.append(self.parent_smooth)
        return smoothings

    def __repr__(self):
        return f"<Rate '{self.name}'>"
