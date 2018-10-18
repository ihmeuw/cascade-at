from cascade.core.parameters import ParameterProperty
from cascade.core.input_data import InputData
from cascade.model.rates import Rate
from cascade.dismod.db.metadata import IntegrandEnum

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class ExecutionContext:
    """
    This is a container for all information about the environment in which a
    model executes. This includes paths to data sources, information about
    cluster resources etc.
    """

    parameters = ParameterProperty()

    def __init__(self):
        self.dismodfile = None


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


class _IntegrandCovariateMultiplier:
    __slots__ = ["name", "value_covariate_multipliers", "std_covariate_multipliers"]

    def __init__(self, name):
        self.name = name
        self.value_covariate_multipliers = []
        self.std_covariate_multipliers = []


class ModelContext:
    __slots__ = ["parameters", "input_data", "rates", "average_integrand_cases", "integrand_covariate_multipliers"]
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
        self.average_integrand_cases = None
        self.integrand_covariate_multipliers = {
            integrand.name: _IntegrandCovariateMultiplier(integrand.name) for integrand in IntegrandEnum
        }
