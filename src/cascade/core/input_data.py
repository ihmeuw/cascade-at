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
        self.constraints = None
        self.covariates = []

    @property
    def ages(self):
        ages = set()
        if self.observations is not None:
            ages.update(self.observations.age_start)
            ages.update(self.observations.age_end)
        if self.constraints is not None:
            ages.update(self.constraints.age_start)
            ages.update(self.constraints.age_end)
        return ages

    @property
    def times(self):
        times = set()
        if self.observations is not None:
            times.update(self.observations.year_start)
            times.update(self.observations.year_end)
        if self.constraints is not None:
            times.update(self.constraints.year_start)
            times.update(self.constraints.year_end)
        return times
