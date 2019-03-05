from pathlib import Path

from rocketsonde.core import Probe, basic_metric

from cascade.core.input_data import InputData
from cascade.core.log import getLoggers
from cascade.core.parameters import ParameterProperty
from cascade.dismod.constants import IntegrandEnum
from cascade.model.rates import Rate

CODELOG, MATHLOG = getLoggers(__name__)


class ExecutionContext:
    """
    This is a container for all information about the environment in which a
    model executes. This includes paths to data sources, information about
    cluster resources etc.
    """
    __slots__ = ["_parameters_parameters", "dismodfile", "resource_monitor"]

    parameters = ParameterProperty()

    def __init__(self):
        self.dismodfile = None
        self.resource_monitor = Probe(basic_metric)

    def db_path(self, location_id):
        """

        Args:
            location_id (int):

        Returns:
            Path: directory in which to write. May not exist.
        """
        if hasattr(self.parameters, "organizational_mode") and self.parameters.organizational_mode == "infrastructure":
            return self.model_base_directory(location_id)
        else:
            if hasattr(self.parameters, "base_directory") and self.parameters.base_directory:
                return (Path(self.parameters.base_directory) / str(location_id)).expanduser()
            else:
                return (Path(".") / str(location_id)).expanduser()

    def model_base_directory(self, location_id):
        if hasattr(self.parameters, "base_directory") and self.parameters.base_directory:
            base_directory = Path(self.parameters.base_directory)
        else:
            base_directory = Path(".")

        if hasattr(self.parameters, "modelable_entity_id") and self.parameters.modelable_entity_id:
            subdir = base_directory / str(self.parameters.modelable_entity_id)
        else:
            subdir = base_directory / "mvid"

        if hasattr(self.parameters, "model_version_id") and self.parameters.model_version_id:
            with_mvid = subdir / str(self.parameters.model_version_id)
        else:
            with_mvid = subdir

        group_locations = location_id // 100
        with_loc = with_mvid / str(group_locations) / str(location_id)
        return with_loc.expanduser()


class _ModelParameters:
    parent_location_id = -1
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
    __slots__ = [
        "parameters", "input_data", "rates", "policies",
        "average_integrand_cases", "integrand_covariate_multipliers"
    ]
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
        self.policies = {}
        self.average_integrand_cases = None
        self.integrand_covariate_multipliers = {
            integrand.name: _IntegrandCovariateMultiplier(integrand.name) for integrand in IntegrandEnum
        }
