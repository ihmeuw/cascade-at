class Density:
    def __init__(self):
        self.density = None
        self.min = None
        self.mean = None
        self.max = None
        self.std = None
        self.eta = None


class DetailDensity(Density):
    def __init__(self):
        super().__init__()
        self.age_lower = None
        self.age_upper = None
        self.time_lower = None
        self.time_upper = None
        self.born_lower = None
        self.born_upper = None


class Defaults:
    def __init__(self):
        self.value = Density()
        self.dage = Density()
        self.dtime = Density()


class MultStd:
    def __init__(self):
        self.value = Density()
        self.dage = Density()
        self.dtime = Density()


