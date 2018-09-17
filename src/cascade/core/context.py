from cascade.core.parameters import ParameterProperty
from cascade.core.input_data import InputData
from cascade.model.rates import Rate


class ExecutionContext:
    """
    This is a container for all information about the environment in which a
    model executes. This includes paths to data sources, information about
    cluster resources etc.
    """

    parameters = ParameterProperty()

    def __init__(self):
        self._dismodfile = None

    @property
    def dismodfile(self):
        return self._dismodfile


class _ModelParameters:
    location_id = -1
    cascade = ParameterProperty()
    node = ParameterProperty()


class _Rates:
    __slots__ = ["pini", "iota", "rho", "chi", "omega"]

    def __init__(self):
        self.pini = Rate("pini")
        self.iota = Rate("iota")
        self.rho = Rate("rho")
        self.chi = Rate("chi")
        self.omega = Rate("omega")

    def __iter__(self):
        return iter([self.pini, self.iota, self.rho, self.chi, self.omega])


class _Outputs:
    __slots__ = ["integrands"]

    def __init__(self):
        self.integrands = _Integrands()


class _Integrand:
    def __init__(self, name):
        self.name = name
        self.grid = None
        self.value_covariate_multipliers = []
        self.std_covariate_multipliers = []


class _Integrands:
    __slots__ = [
        "Sincidence",
        "remission",
        "mtexcess",
        "mtother",
        "mtwith",
        "susceptible",
        "withC",
        "prevalence",
        "Tincidence",
        "mtspecific",
        "mtall",
        "mtstandard",
        "relrisk",
    ]

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, _Integrand(name))

    def __iter__(self):
        for name in self.__slots__:
            yield getattr(self, name)


class ModelContext:
    """
    This is a container for all inputs, parametrization and data, necessary
    to run the model for a node in the hierarchy. It does not include any
    information about the computational environment in which the model is
    running, no file paths or cluster information.
    """

    def __init__(self):
        self.parameters = _ModelParameters()
        self.input_data = InputData()
        self.rates = _Rates()
        self.outputs = _Outputs()
