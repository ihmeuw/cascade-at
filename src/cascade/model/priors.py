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

    def __init__(self, lower, upper, mean=None, name=None):
        super().__init__(name=name)
        if mean is None:
            mean = (upper - lower) / 2 + lower
        _validate_bounds(lower, mean, upper)

        self.lower = lower
        self.upper = upper
        self.mean = mean

    def parameters(self):
        return {"lower": self.lower, "upper": self.upper, "mean": self.mean}


class ConstantPrior:
    density = "uniform"

    def __init__(self, value, name=None):
        super().__init__(name)
        self.value = value

    def parameters(self):
        return {"lower": self.value, "upper": self.value, "mean": self.value}


class GaussianPrior(_Prior):
    density = "gaussian"

    def __init__(self, mean, standard_deviation, lower=float("-inf"), upper=float("inf"), name=None):
        super().__init__(name=name)
        _validate_bounds(lower, mean, upper)
        _validate_standard_deviation(standard_deviation)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation

    def _parameters(self):
        return {"lower": self.lower, "upper": self.upper, "mean": self.mean, "std": self.standard_deviation}


class LaplacePrior(GaussianPrior):
    density = "laplace"


class StudentsTPrior(_Prior):
    density = "students"

    def __init__(self, mean, standard_deviation, nu, lower=float("-inf"), upper=float("inf"), name=None):
        super().__init__(name=name)
        _validate_bounds(lower, mean, upper)
        _validate_standard_deviation(standard_deviation)
        _validate_nu(nu)

        self.lower = lower
        self.upper = upper
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.nu = nu

    def _parameters(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
            "nu": self.nu,
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
