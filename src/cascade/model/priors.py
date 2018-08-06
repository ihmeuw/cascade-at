class Prior:
    def __init__(self, name=None):
        self.name = name

    def parameters(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(frozenset(self.parameters().items()))

    def __eq__(self, other):
        return isinstance(other, Prior) and self.name == other.name and self.parameters() == other.parameters()


class UniformPrior(Prior):
    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value

    def parameters(self):
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

    def parameters(self):
        return {
            "density": "gaussian",
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.standard_deviation,
        }
