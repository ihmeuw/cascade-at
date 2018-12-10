from cascade.core.db import dbtrees

from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def get_location_hierarchy_from_gbd(execution_context):
    return dbtrees.loctree(location_set_id=35, gbd_round_id=execution_context.parameters.gbd_round_id)


def get_descendents(execution_context, children_only=False, include_parent=False):
    location_hierarchy = get_location_hierarchy_from_gbd(execution_context)
    location = location_hierarchy.get_node_by_id(execution_context.parameters.location_id)

    if children_only:
        descendents = location.children
    else:
        descendents = location.all_descendants()

    if include_parent:
        nodes = descendents + [location]
    else:
        nodes = descendents

    return [d.id for d in nodes]
