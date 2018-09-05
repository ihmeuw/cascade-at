from functools import total_ordering

# TODO: Several of these prior types accept eta as a parameter and the others
# don't use it as a parameter but use it as an offset in the optimization
# process only if certain conditions are met. I think that means that the
# offset case should be represented differently and there is also a validation
# of eta for that case which could be done but isn't. For more details see the
# docs: https://bradbell.github.io/dismod_at/doc/prior_table.htm#eta.Scaling%20Fixed%20Effects


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

    def __hash__(self):
        return hash((frozenset(self.parameters().items()), self.name))

    def __eq__(self, other):
        return isinstance(other, _Prior) and self.name == other.name and self.parameters() == other.parameters()

    def __lt__(self, other):
        if not isinstance(other, _Prior):
            return NotImplemented
        self_dict = sorted([(k, v) for k, v in dict(name=self.name, **self.parameters()).items() if v is not None])
        other_dict = sorted([(k, v) for k, v in dict(name=other.name, **other.parameters()).items() if v is not None])

        return self_dict < other_dict

    def __repr__(self):
        return f"<{type(self).__name__} {self.parameters()}>"


def _validate_bounds(lower, mean, upper):
    if not lower <= mean <= upper:
        raise ValueError("Bounds are inconsistent")


def _validate_standard_deviation(standard_deviation):
    if standard_deviation < 0:
        raise ValueError("Standard deviation must be positive")


def _validate_nu(nu):
    if nu < 0:
        raise ValueError("Nu must be positive")


class UniformPrior(_Prior):
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


class ConstantPrior(_Prior):
    density = "uniform"

    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value

    def _parameters(self):
        return {"lower": self.value, "upper": self.value, "mean": self.value}


class GaussianPrior(_Prior):
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


class LaplacePrior(GaussianPrior):
    density = "laplace"


class StudentsTPrior(_Prior):
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


class LogGaussianPrior(_Prior):
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


class LogLaplacePrior(LogGaussianPrior):
    density = "log_laplace"


class LogStudentsTPrior(_Prior):
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

NO_PRIOR = UniformPrior(float("-inf"), float("inf"), 0)
ZERO = UniformPrior(0, 0, 0)
ZERO_TO_ONE = UniformPrior(0, 1, 0.1)
MINUS_ONE_TO_ONE = UniformPrior(-1, 1, 0)
