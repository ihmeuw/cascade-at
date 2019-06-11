from collections import Counter
from collections.abc import Mapping

from cascade.dismod.constants import WeightEnum
from cascade.model.covariates import Covariate
from cascade.model.dismod_groups import DismodGroups
from cascade.model.var import Var
from cascade.model.smooth_grid import uninformative_grid_from_var


class Model(DismodGroups):
    """
    A DismodGroups container of SmoothGrid.

    Uses locations as given and translates them into nodes for Dismod-AT.
    Uses ages and times as given and translates them into ``age_id``
    and ``time_id`` for Dismod-AT.
    """
    def __init__(self, nonzero_rates, parent_location, child_location=None, covariates=None, weights=None):
        """
        >>> locations = location_hierarchy(6, location_set_version_id=429)
        >>> m = Model(["chi", "omega", "iota"], 6, locations.successors(6))

        Args:
            nonzero_rates (List[str]): A list of rates, using the Dismod-AT
                terms for the rates, so they are "iota", "chi", "omega",
                "rho", and "pini".
            parent_location (int): The location ID for the parent.
            child_location (List[int]): List of the children.
            covariates (List[Covariate]): A list of covariate objects.
                This supplies the reference values and max differences,
                used to exclude data by covariate value.
            weights (Dict[str,Var]): There are four kinds of weights:
                "constant", "susceptible", "with_condition", and "total".
                No other weights are used.
        """
        super().__init__()
        self.nonzero_rates = nonzero_rates
        self.location_id = parent_location
        self.child_location = child_location if child_location else list()
        double_locations = [l for (l, v) in Counter(self.child_location + [parent_location]).items() if v > 1]
        if double_locations:
            raise ValueError(f"Multiple locations have same ID {double_locations}")
        # Covariates are here because their reference values are part of
        # the model. Even though avgint and data use them, a model is always
        # written before the avgint and data are written.
        self.covariates = covariates if covariates else list()
        assert isinstance(self.covariates, list)
        if len(self.covariates) > 0:
            assert isinstance(self.covariates[0], Covariate)
        self._check_covariates(self.covariates)
        # There are always four weights, constant, susceptible,
        # with_condition, and total.
        if weights:
            self.weights = weights
        else:
            self.weights = dict()
        self._check_weights(self.weights)
        self._scale = None
        self.scale_set_by_user = False

    @property
    def scale(self):
        """The scale is a Var, so it has a value for every model variable.
        It is the value of the model variable at which to evaluate the
        derivative of its base log-likelihood. This derivative sets the
        baseline against which nonlinear optimization will compare later
        values. If the derivative is zero, then the optimization will ignore
        this variable.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        """Dismod-AT usually calculates this for you. If you've set it by
        hand, this records that this is the case so that it will rewrite
        what Dismod-AT calculates."""
        self.scale_set_by_user = True
        self._scale = value

    def model_like(self):
        """Make another model with the same structure as this one but
        without priors."""
        model = Model(self.nonzero_rates, self.location_id, self.child_location,
                      self.covariates, self.weights)
        if self.scale_set_by_user:
            model._scale = self._scale
        return model

    def write(self, writer):
        self._ensure_weights()
        self._check()
        writer.start_model(self.nonzero_rates, self.child_location)
        for group in self.values():
            for grid in group.values():
                writer.write_ages_and_times(grid.ages, grid.times)
        for weight_value in self.weights.values():
            writer.write_ages_and_times(weight_value.ages, weight_value.times)

        writer.write_covariate(self.covariates)
        writer.write_weights(self.weights)
        for group_name, group in self.items():
            if group_name == "rate":
                for rate_name, grid in group.items():
                    writer.write_rate(rate_name, grid)
            elif group_name == "random_effect":
                for (rate_name, child), grid in group.items():
                    writer.write_random_effect(rate_name, child, grid)
            elif group_name in {"alpha", "beta", "gamma"}:
                for (covariate, target), grid in group.items():
                    writer.write_mulcov(group_name, covariate, target, grid)
            else:
                raise RuntimeError(f"Unknown kind of field {group_name}")

    def var_from_mean(self):
        # Call the mean mu because mean is a function.
        mu = DismodGroups()
        for group_name, group in self.items():
            if group_name != "random_effect":
                for key, grid in group.items():
                    mu[group_name][key] = grid.var_from_mean()
            else:
                for key, grid in group.items():
                    # One Random Effect grid creates many child vars.
                    if key[1] is None:
                        for child in self.child_location:
                            mu[group_name][(key[0], child)] = grid.var_from_mean()
                    else:
                        mu[group_name][key] = grid.var_from_mean()
        return mu

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.scale_set_by_user and not (other.scale_set_by_user and self._scale == other._scale):
            return False
        return (set(self.nonzero_rates) == set(other.nonzero_rates) and
                self.location_id == other.location_id and
                set(self.child_location) == set(other.child_location) and
                set(self.covariates) == set(other.covariates) and
                self.weights == other.weights
                )

    def _ensure_weights(self):
        """If weights weren't made, then make them constant. Must be done after
        there is data in the Model."""
        # Find an age and time already in the model because adding an
        # age and time outside the model can change the integration ranges.
        arbitrary_grid = next(iter(self.rate.values()))
        one_age_time = (arbitrary_grid.ages[0:1], arbitrary_grid.times[0:1])

        for kind in (weight.name for weight in WeightEnum):
            if kind not in self.weights:
                self.weights[kind] = Var(*one_age_time)
                self.weights[kind].grid.loc[:, "mean"] = 1.0

    def _check(self):
        child_specific_rate = dict()
        for rate, child in self.random_effect:
            child_specific = child is not None
            if rate in child_specific_rate and child_specific_rate[rate] != child_specific:
                raise ValueError(
                    f"Model random effect for {rate} has both child-specific "
                    f"and all-child specifications")
            else:
                child_specific_rate[rate] = child_specific

    @staticmethod
    def _check_covariates(covariates):
        for c in covariates:
            if not isinstance(c, Covariate):
                raise TypeError(f"Covariate passed to model isn't an instance of covariate {c}.")

    @staticmethod
    def _check_weights(weights):
        if not isinstance(weights, Mapping):
            raise TypeError(f"Weights are a dictionary from string to Var classes, not {type(weights)}.")
        for name, weight in weights.items():
            if not isinstance(weight, Var):
                raise TypeError(f"Each weight should be a Var object, not {name}={type(weight)}.")
            if name not in dir(WeightEnum):
                raise ValueError(f"Weights should be one of {[w.name for w in WeightEnum]}")

    @classmethod
    def from_var(cls, var, parent_location, weights=None, covariates=None,
                 multiple_random_effects=False):
        """
        Given values across all rates, construct a model with loose priors
        in order to be able to predict from those rates.

        Args:
            var (DismodGroups[Var]): Values on grids.
            parent_location (int): A parent location, because that isn't
                in the keys.
            weights (Dict[Var]): Population weights for integrands.
            covariates(List[Covariate]): Covariate objects.
            multiple_random_effects (bool): Create a separate smooth grid for
                each random effect in the Var. The default is to create a
                single smooth grid for all random effects.

        Returns:
            Model: with Uniform distributions everywhere and no mulstd.
        """
        child_locations = list(sorted({k[1] for k in var.random_effect.keys()}))
        nonzero_rates = list(var.rate.keys())
        model = cls(nonzero_rates, parent_location, child_locations, weights=weights, covariates=covariates)

        strictly_positive = dict(rate=True)
        for group_name, group in var.items():
            skip_re_children = group_name == "random_effect" and not multiple_random_effects
            for key, var in group.items():
                if skip_re_children:
                    # Make one smooth grid for all children.
                    assign_key = (key[0], None)
                else:
                    assign_key = key

                if assign_key not in model[group_name]:
                    must_be_positive = strictly_positive.get(group_name, False)
                    model[group_name][assign_key] = uninformative_grid_from_var(var, must_be_positive)

        return model
