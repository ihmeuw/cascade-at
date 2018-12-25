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
from cascade.model.priors import _Prior


PRIOR_KINDS = ["value", "dage", "dtime"]


class RandomField:
    def __init__(self, age_time_grid, grid_priors=None):
        """

        Args:
            age_time_grid (AgeTimeGrid): The supporting grid.
            priors (dict[str,PriorGrid]): Three priors at every grid point.
        """
        self.age_time_grid = age_time_grid
        self.grid_priors = grid_priors


class PartsContainer:
    def __init__(self, nonzero_rates, child_location):
        self.rate = dict(zip(nonzero_rates, repeat(None)))
        self.random_effect = dict(zip(product(nonzero_rates, child_location), repeat(None)))
        self.alpha = dict()
        self.beta = dict()
        self.gamma = dict()


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
        self.covariates = dict()  # from name to Covariate

        self.parts = PartsContainer(nonzero_rates, self.child_location)

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
