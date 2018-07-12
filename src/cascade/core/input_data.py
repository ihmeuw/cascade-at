class Rates:
    def __init__(self):
        self.incidence = None
        self.excess_mortality = None


class InputData:
    """Container for all the input daat necessary to run a model.

    TODO: This is missing a bunch of slots which we will add in incrementally.
    """

    def __init__(self):
        self.rates = Rates()
