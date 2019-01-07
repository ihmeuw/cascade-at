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
from itertools import product
from math import nan

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import DensityEnum


PRIOR_KINDS = ["value", "dage", "dtime"]


class PriorView:
    """Slices to access priors with Distribution objects."""
    def __init__(self, parent, kind):
        self._kind = kind
        self._parent = parent
        self.param_names = ["density_id", "mean", "std", "lower", "upper", "eta", "nu"]

    def __setitem__(self, at_slice, value):
        """
        Args:
            at_slice (slice, slice): What to change, as integer offset into ages and times.
            value (priors.Prior): The prior to set, containing dictionary of
                                  parameters.
        """
        ages = self._parent.ages[at_slice[0]]
        times = self._parent.times[at_slice[1]]
        to_set = value.parameters()
        to_set["density_id"] = DensityEnum[to_set["density"]].value
        self._parent.priors.loc[
            np.in1d(self._parent.priors.age, ages) & np.in1d(self._parent.priors.time, times)
            & (self._parent.priors.kind == self._kind),
            self.param_names
        ] = [to_set[setp] if setp in to_set else nan for setp in self.param_names]


class RandomField:
    def __init__(self, age_time_grid):
        """
        Args:
            age_time_grid (Tuple(set,set)): The supporting grid.
        """
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        self.parameter_columns = ["density_id", "mean", "lower", "upper", "std", "nu", "eta"]
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        one_priors = pd.DataFrame(dict(
            prior_name=None,
            kind="value",
            age=age_time[:, 0],
            time=age_time[:, 1],
            density_id=nan,
            mean=nan,
            lower=nan,
            upper=nan,
            std=nan,
            nu=nan,
            eta=nan,
        ))
        one_priors = one_priors.append({"age": nan, "time": nan, "kind": "value", "name": None}, ignore_index=True)
        self.priors = pd.concat([one_priors, one_priors.assign(kind="dage"), one_priors.assign(kind="dtime")])
        self.priors = self.priors.reset_index(drop=True)

    @property
    def value(self):
        return PriorView(self, "value")

    @property
    def dage(self):
        return PriorView(self, "dage")

    @property
    def dtime(self):
        return PriorView(self, "dtime")


class FieldDraw:
    def __init__(self, age_time_grid, count=1):
        self.ages = np.array(age_time_grid[0], dtype=np.float)
        self.times = np.array(age_time_grid[1], dtype=np.float)
        age_time = np.array(list(product(sorted(self.ages), sorted(self.times))))
        self.values = pd.DataFrame(dict(
            age=np.tile(age_time[:, 0], count),
            time=np.tile(age_time[:, 1], count),
            mean=nan,
            idx=np.repeat(range(count), len(age_time)),
        ))
        self.mulstd = pd.DataFrame(dict(
            kind=np.tile(["value", "age", "time"], count),
            idx=np.repeat(range(count), 3),
            mean=nan,
        ))


class PartsContainer:
    """The Model has five kinds of random fields, but if we take the
    vars from the model, the vars split into the same five kinds. This class
    is responsible for organizing data into those five kinds and iterating
    over them, for a Model or for vars, or for priors, whatever."""
    def __init__(self):
        # Key is the rate as a string.
        self.rate = dict()
        # Key is tuple (rate, location_id)  # location_id=None means no nslist.
        self.random_effect = dict()
        # Key is (covariate, rate), both as strings.
        self.alpha = dict()
        # Key is (covariate, integrand), both as strings.
        self.beta = dict()
        # Key is (covariate, integrand), both as strings.
        self.gamma = dict()

    def items(self):
        for part_name in ["rate", "random_effect", "alpha", "beta", "gamma"]:
            for k, v in getattr(self, part_name).items():
                if isinstance(k, str):
                    yield [part_name, k], v
                else:
                    yield tuple([part_name] + list(k)), v

    def values(self):
        for part_name in ["rate", "random_effect", "alpha", "beta", "gamma"]:
            yield from getattr(self, part_name).values()


class Model:
    def __init__(self, nonzero_rates, child_location):
        """
        >>> locations = location_hierarchy(execution_context)
        >>> m = Model(["chi", "omega", "iota"], locations, 6)
        """
        self.nonzero_rates = nonzero_rates
        self.child_location = child_location
        self.covariates = list()  # of class Covariate
        self.weights = dict()

        self.parts = PartsContainer()

    def write(self, writer):
        writer.start_model(self.nonzero_rates, self.child_location)
        for field_at in self.parts.values():
            writer.write_ages_and_times(field_at.ages, field_at.times)
        for weight_value in self.weights.values():
            writer.write_ages_and_times(weight_value.ages, weight_value.times)

        writer.write_covariate(self.covariates)
        for name, weight in self.weights.items():
            writer.write_weight(name, weight)

        for kind, write_field in self.parts.items():
            if kind[0] == "rate":
                writer.write_rate(kind[1], write_field)
            elif kind[0] == "random_effect":
                writer.write_random_effect(kind[1], kind[2], write_field)
            elif kind[0] == "alpha":
                writer.write_mulcov("alpha", kind[1], kind[2], write_field)
            elif kind[0] == "beta":
                writer.write_mulcov("beta", kind[1], kind[2], write_field)
            elif kind[0] == "gamma":
                writer.write_mulcov("gamma", kind[1], kind[2], write_field)
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
        return PartsContainer()
