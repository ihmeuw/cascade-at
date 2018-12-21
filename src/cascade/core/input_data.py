from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class Rates:
    def __init__(self):
        self.incidence = None
        self.excess_mortality = None


class InputData:
    """Container for all the input data necessary to run a model.

    TODO: This is missing a bunch of slots which we will add in incrementally.
    """

    def __init__(self):
        self.observations = None
        self.covariates = []

    @property
    def ages(self):
        ages = set()
        if self.observations is not None:
            ages.update(self.observations.age_lower)
            ages.update(self.observations.age_upper)
        return ages

    @property
    def times(self):
        times = set()
        if self.observations is not None:
            times.update(self.observations.time_lower)
            times.update(self.observations.time_upper)
        return times
