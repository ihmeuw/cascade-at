from .parameters import ParameterProperty
from .input_data import InputData
from cascade.model.rates import Rate


class ExecutionContext:
    """
    This is a container for all information about the environment in which a
    model executes. This includes paths to data sources, information about
    cluster resources etc.
    """

    parameters = ParameterProperty()


class _ModelParameters:
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
