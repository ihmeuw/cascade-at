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


def location_id_from_location_and_level(execution_context, location_id, target_level):
    """ Find the location which is above location_id at the target level in the hierarchy

    Args:
    location_id: the location to search up from
    target_level: A level in the hierarchy where 1==global and larger numbers are more detailed
                  and the string "most_detailed" indicates the most detailed level.

    Raises:
    ValueError if location_id is itself above target_level in the hierarchy
    """
    hierarchy = get_location_hierarchy_from_gbd(execution_context)
    node = hierarchy.get_node_by_id(location_id)

    if target_level == "most_detailed":
        if node.children:
            raise ValueError("Most detailed level selected but current location is higher in the hierarchy than that")
    else:
        target_level = int(target_level)

        # The -1 here is because epiviz uses a system where global == 1 and
        # central comp uses a system where global == 0
        normalized_target = target_level - 1

        while hierarchy.get_nodelvl_by_id(node.id) > normalized_target:
            node = node.parent
        if hierarchy.get_nodelvl_by_id(node.id) != normalized_target:
            level_name = {1: "Global", 2: "Super Region", 3: "Region", 4: "Country", 5: "Subnational 1"}[target_level]
            raise ValueError(f"Level '{level_name}' selected but current location is higher in the hierarchy than that")

    return node.id
