from cascade.core.db import dbtrees

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def get_location_hierarchy_from_gbd(execution_context):
    return dbtrees.loctree(location_set_id=35, gbd_round_id=execution_context.parameters.gbd_round_id)
