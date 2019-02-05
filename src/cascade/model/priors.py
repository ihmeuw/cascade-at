from copy import copy
from functools import total_ordering

import numpy as np

from cascade.core import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

# A description of how dismod interprets these distributions and their parameters can be found here:
# https://bradbell.github.io/dismod_at/doc/prior_table.htm


class PriorError(Exception):
    pass


@total_ordering
class _Prior:
    """The base for all Priors
    """

    density = None

    def __init__(self, name=None):
        self.name = name

    def _parameters(self):
        raise NotImplementedError()

    def parameters(self):
        return dict(density=self.density, **self._parameters())

    def assign(self, **kwargs):
        """Create a new distribution with modified parameters."""
        modified = copy(self)
        if set(kwargs.keys()) - set(self.__dict__.keys()):
            missing = list(sorted(set(kwargs.keys()) - set(self.__dict__.keys())))
            raise AttributeError(f"The prior doesn't have these attributes {missing}.")
        modified.__dict__.update(kwargs)
        return modified

    def __hash__(self):
        return hash((frozenset(self.parameters().items()), self.name))

    def __eq__(self, other):
        if not isinstance(other, _Prior):
            return NotImplemented
        return self.name == other.name and self.parameters() == other.parameters()

    def __lt__(self, other):
        if not isinstance(other, _Prior):
            return NotImplemented
        self_dict = sorted([(k, v) for k, v in dict(name=self.name, **self.parameters()).items() if v is not None])
        other_dict = sorted([(k, v) for k, v in dict(name=other.name, **other.parameters()).items() if v is not None])

        return self_dict < other_dict

    def __repr__(self):
        return f"<{type(self).__name__} {self.parameters()}>"


def _validate_bounds(lower, mean, upper):
    any_nones = lower is None or mean is None or upper is None
    any_invalid = any_nones or np.isnan(lower) or np.isnan(mean) or np.isnan(upper)
    if any_invalid:
        raise PriorError(f"Bounds contain invalid values: lower={lower} mean={mean} upper={upper}")
    if not lower <= mean <= upper:
        raise PriorError(f"Bounds are inconsistent: lower={lower} mean={mean} upper={upper}")


def _validate_standard_deviation(standard_deviation):
    if standard_deviation is None or np.isnan(standard_deviation) or standard_deviation < 0:
        raise PriorError(f"Standard deviation must be positive: standard deviation={standard_deviation}")


def _validate_nu(nu):
    if nu is None or np.isnan(nu) or nu < 0:
        raise PriorError(f"Nu must be positive: nu={nu}")


class Uniform(_Prior):
    density = "uniform"

    def __init__(self, lower, upper, mean=None, eta=None, name=None):
        super().__init__(name=name)
        if mean is None:
            mean = (upper + lower) / 2
        _validate_bounds(lower, mean, upper)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.eta = eta

    def _parameters(self):
        return {"lower": self.lower, "upper": self.upper, "mean": self.mean, "eta": self.eta}


class Constant(_Prior):
    density = "uniform"

    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value

    def _parameters(self):
        return {"lower": self.value, "upper": self.value, "mean": self.value}


class Gaussian(_Prior):
    density = "gaussian"

    def __init__(self, mean, standard_deviation, lower=float("-inf"), upper=float("inf"), eta=None, name=None):
        super().__init__(name=name)
        _validate_bounds(lower, mean, upper)
        _validate_standard_deviation(standard_deviation)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.eta = eta

    def _parameters(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
            "eta": self.eta,
        }


class Laplace(Gaussian):
    density = "laplace"


class StudentsT(_Prior):
    density = "students"

    def __init__(self, mean, standard_deviation, nu, lower=float("-inf"), upper=float("inf"), eta=None, name=None):
        super().__init__(name=name)
        _validate_bounds(lower, mean, upper)
        _validate_standard_deviation(standard_deviation)
        _validate_nu(nu)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.nu = nu
        self.eta = eta

    def _parameters(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
            "nu": self.nu,
            "eta": self.eta,
        }


class LogGaussian(_Prior):
    density = "log_gaussian"

    def __init__(self, mean, standard_deviation, eta, lower=float("-inf"), upper=float("inf"), name=None):
        super().__init__(name=name)
        _validate_bounds(lower, mean, upper)
        _validate_standard_deviation(standard_deviation)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.eta = eta

    def _parameters(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
            "eta": self.eta,
        }


class LogLaplace(LogGaussian):
    density = "log_laplace"


class LogStudentsT(_Prior):
    density = "log_students"

    def __init__(self, mean, standard_deviation, nu, eta, lower=float("-inf"), upper=float("inf"), name=None):
        super().__init__(name=name)
        _validate_bounds(lower, mean, upper)
        _validate_standard_deviation(standard_deviation)
        _validate_nu(nu)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.nu = nu
        self.eta = eta

    def _parameters(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
            "nu": self.nu,
            "eta": self.eta,
        }


# Useful predefined priors

NO_PRIOR = Uniform(float("-inf"), float("inf"), 0, name="null_prior")
ZERO = Uniform(0, 0, 0, name="constrain_to_zero")
ZERO_TO_ONE = Uniform(0, 1, 0.1, name="uniform_zero_to_one")
MINUS_ONE_TO_ONE = Uniform(-1, 1, 0, name="uniform_negative_one_to_one")


DENSITY_ID_TO_PRIOR = {
    0: Uniform,
    1: Gaussian,
    2: Laplace,
    3: StudentsT,
    4: LogGaussian,
    5: LogLaplace,
    6: LogStudentsT,
}


def prior_distribution(parameters):
    density, lower, upper, value, stdev, eta, nu = [
        parameters[name] for name in
        [
            "density", "lower", "upper", "mean", "std", "eta", "nu"
        ]
    ]
    if np.isclose(lower, upper):
        return Constant(value)
    elif density == "uniform":
        return Uniform(lower, upper, value, eta)
    elif density == "gaussian":
        return Gaussian(value, stdev, lower, upper, eta)
    elif density == "laplace":
        return Laplace(value, stdev, lower, upper, eta)
    elif density == "students":
        return StudentsT(value, stdev, nu, lower, upper, eta)
    elif density == "log_gaussian":
        return LogGaussian(value, stdev, eta, lower, upper)
    elif density == "log_laplace":
        return LogLaplace(value, stdev, eta, lower, upper)
    elif density == "log_students":
        return LogStudentsT(value, stdev, nu, eta, lower, upper)
    else:
        return None
