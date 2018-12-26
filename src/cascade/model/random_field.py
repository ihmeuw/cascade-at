"""
The model is a collection of random fields plus some specification
for the solver.

All the kinds of objects:

 * Random fields in a model.
   * age-time grid
   * priors
     * prior distribution and prior parameters on a grid
     * prior distribution and prior parameters of stdev multipliers
   * vars, which are outcomes of several types
     * initial guess
     * fit
     * truth
     * samples, where there are more than one.
 * Measurement data

"""
from itertools import product, repeat

from cascade.model.grids import AgeTimeGrid, PriorGrid


PRIOR_KINDS = ["value", "dage", "dtime"]


class RandomField:
    def __init__(self, age_time_grid, priors=None):
        """
        Args:
            age_time_grid (AgeTimeGrid): The supporting grid.
            priors (dict[str,PriorGrid]): Three priors at every grid point.
                                          Includes mulstd priors.
        """
        assert isinstance(age_time_grid, AgeTimeGrid)
        for key, value in priors.items():
            assert key in ["value", "dage", "dtime"]
            assert isinstance(value, PriorGrid)

        self.age_time_grid = age_time_grid
        self.priors = priors


class PartsContainer:
    """The Model has five kinds of random fields, but if we take the
    vars from the model, the vars split into the same five kinds. This class
    is responsible for organizing data into those five kinds and iterating
    over them, for a Model or for vars, or for priors, whatever."""
    def __init__(self, nonzero_rates, child_location):
        # Key is the rate as a string.
        self.rate = dict(zip(nonzero_rates, repeat(None)))
        # Key is tuple (rate, location_id)
        self.random_effect = dict(zip(product(nonzero_rates, child_location), repeat(None)))
        # Key is (covariate, rate), both as strings.
        self.alpha = dict()
        # Key is (covariate, integrand), both as strings.
        self.beta = dict()
        # Key is (covariate, integrand), both as strings.
        self.gamma = dict()

    def by_field(self):
        for k, v in self.rate:
            yield "rate", k, v
        for k, v in self.random_effect:
            yield "random_effect", k, v
        for k, v in self.alpha:
            yield "alpha", k, v
        for k, v in self.beta:
            yield "beta", k, v
        for k, v in self.gamma:
            yield "gamma", k, v


class Model:
    def __init__(self, nonzero_rates, locations, parent_location):
        """
        >>> locations = location_hierarchy(execution_context)
        >>> m = Model(["chi", "omega", "iota"], locations, 6)
        """
        self.nonzero_rates = nonzero_rates
        self.locations = locations
        self.parent_location = parent_location
        self.child_location = set(locations.successors(parent_location))
        self.covariates = list()  # of class Covariate

        self.parts = PartsContainer(nonzero_rates, self.child_location)

    def write(self, writer):
        writer.start_model(self.nonzero_rates, self.parent_location, self.child_location)
        for which, field_at in self.parts.by_field():
            at_grid = field_at.age_time_grid
            writer.write_ages_and_times(at_grid.ages, at_grid.times)
        writer.write_covariate(self.covariates)
        writer.write_locations(self.locations)

        for kind, key, write_field in self.parts.by_field():
            if kind == "rate":
                writer.write_rate(key, write_field)
            elif kind == "random_effect":
                writer.write_random_effect(key, write_field)
            elif kind == "alpha":
                writer.write_mulcov("alpha", key, write_field)
            elif kind == "beta":
                writer.write_mulcov("beta", key, write_field)
            elif kind == "gamma":
                writer.write_mulcov("gamma", key, write_field)
            else:
                raise RuntimeError(f"Unknown kind of field {kind}")

    @property
    def rate(self):
        return self.parts.rate

    @property
    def random_effect(self):
        return self.parts.random_effect

    @property
    def alpha(self):
        return self.parts.alpha

    @property
    def beta(self):
        return self.parts.beta

    @property
    def gamma(self):
        return self.parts.gamma

    @property
    def model_variables(self):
        return PartsContainer(self.nonzero_rates, self.child_location)
