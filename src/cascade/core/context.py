from pathlib import Path

from rocketsonde.core import Probe, basic_metric

from cascade.core.log import getLoggers
from cascade.core.parameters import ParameterProperty

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

    def model_base_directory(self, location_id, sex=None):
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
        if sex:
            with_loc = with_loc / sex
        return with_loc.expanduser()
