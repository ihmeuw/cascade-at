import networkx as nx
from numpy import nan
import pandas as pd

from cascade.core.db import db_queries
from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def location_hierarchy(gbd_round_id, location_set_version_id=None, location_set_id=None):
    """
    The GBD location hierarchy as a networkx graph where each node is the
    location id, and its properties are all properties returned by
    the dbtrees library. For instance

    >>> locations = location_hierarchy(6, location_set_id=35)
    >>> locations = location_hierarchy(6, location_set_version_id=429)
    >>> assert locations.nodes[1]["level"] == 0
    >>> assert locations.nodes[13]["location_name"] == "Malaysia"
    >>> assert locations.successors(5) == [6, 7, 8]
    >>> assert locations.predecessors(491) == [6]

    Args:
        execution_context: Uses the ``gbd_round_id``.

    Returns:
        nx.DiGraph: Each node is the location id as an integer.
    """
    choose_by = dict()
    if location_set_id:
        choose_by["location_set_id"] = location_set_id
    elif location_set_version_id:
        choose_by["location_set_version_id"] = location_set_version_id
    else:
        raise TypeError(f"Must specify either location set version or location set.")

    location_df = db_queries.get_location_metadata(
        gbd_round_id=gbd_round_id, **choose_by)
    G = nx.DiGraph()
    G.add_nodes_from([(int(row.location_id), row._asdict()) for row in location_df.itertuples()])
    # GBD encodes the global node as having itself as a parent.
    G.add_edges_from([(int(row.parent_id), int(row.location_id))
                      for row in location_df[location_df.location_id != 1].itertuples()])
    G.graph["root"] = 1  # Global is the root location_id
    return G


def location_hierarchy_to_dataframe(locations):
    """Converts the tree of locations into a Pandas Dataframe suitable
    for passing to a Session.

    Args:
        locations (nx.DiGraph): A locations, as returned by
            :py:function:`cascade.input_data.db.locations.location_hierarchy`_.
    """
    sorted_locations = list(nx.lexicographical_topological_sort(locations))
    parents = list()
    names = list()
    for l in sorted_locations:
        parent = list(locations.predecessors(l))
        if parent:
            parents.append(parent[0])
        else:
            parents.append(nan)
        names.append(locations.nodes[l]["location_name"])
    return pd.DataFrame(dict(
        location_id=sorted_locations,
        parent_id=parents,
        name=names,
    ))


def get_descendants(locations, location_id, children_only=False, include_parent=False):
    """
    Retrieves a parent and direct children, or all descendants, or not the
    parent.

    Args:
        locations: Graph of locations from ``location_hierarchy``
        location_id: The location for which to get descendants.
        children_only (bool): Exclude children of the children and below.
        include_parent (bool): Add the parent location to return results.

    Returns:
        set of location IDs
    """
    if children_only:
        nodes = set(locations.successors(location_id))
    else:
        nodes = nx.descendants(locations, location_id)

    if include_parent:
        nodes.add(location_id)
    elif location_id in nodes:
        nodes.remove(location_id)
    # don't include parent and parent isn't in there, so OK.

    return list(nodes)


def all_locations_with_these_parents(all_locations, subset_of_locations):
    """
    Given a subset of locations, find all children. So if
    the US is a location and WA is a location, include all other states
    and all counties in WA.
    """
    total = set(subset_of_locations)
    for find_children in subset_of_locations:
        total |= set(all_locations.successors(find_children))
    return list(total)


def location_id_from_start_and_finish(locations, start, finish):
    """ Find the set of locations from a parent to a child.

    Args:
        locations: Graph of locations.
        start (int): location ID at which to start the cascade
        finish (int): location ID of a descendant of the start, possibly
                      the same location ID as the start, itself.

    Returns:
        List[int]: The list of locations from the drill start to the given
                   location.

    Raises:
        ValueError if finish isn't a descendant of the start.
    """
    if start is not None:
        start = int(start)
    finish = int(finish)
    try:
        drill_nodes = nx.ancestors(locations, finish) | {finish}
    except nx.NetworkXError as nxe:
        raise ValueError(f"Location {finish} isn't in the location set {list(locations.nodes)}.") from nxe
    drill_to_top = list(nx.topological_sort(nx.subgraph(locations, nbunch=drill_nodes)))
    if start:
        try:
            drill = drill_to_top[drill_to_top.index(start):]
        except ValueError as ve:
            raise ValueError(f"Location {start} isn't an ancestor of location {finish}.") from ve
    else:
        drill = drill_to_top
    return drill


def location_id_from_location_and_level(locations, location_id, target_level):
    """ Find the set of locations from the destination location to
    the ``target_level`` above that location.

    Args:
        locations: Graph of locations.
        location_id (int): the location to search up from
        target_level (str,int): A level in the hierarchy where 1==global and larger numbers are more detailed
                      and the string "most_detailed" indicates the most detailed level.
                      Must be 1 or greater.

    Returns:
        List[int]: The list of locations from the drill start to the given
                   location.

    Raises:
        ValueError if location_id is itself above target_level in the hierarchy

    NOTE:
        This makes some assumptions about what level will be selectable in epiviz which could change
        in the future. There may be a more future-proof version of this that loads stuff out of the
        epi.cascade_level table instead of hard coding it.
    """
    if target_level == "most_detailed":
        if not list(locations.successors(location_id)):
            return [location_id]
        else:
            raise ValueError(f"Most detailed level selected but location {location_id} has child locations "
                             f"{list(locations.successors(location_id))}")

    else:
        drill_nodes = nx.ancestors(locations, location_id) | {location_id}
        drill = list(nx.topological_sort(nx.subgraph(locations, nbunch=drill_nodes)))
        target_level = int(target_level)
        if target_level <= 0:
            raise ValueError(
                f"Expected a location level greater than 0 but found {target_level}")

        # The -1 here is because EpiViz uses a system where global == 1 and
        # central comp uses a system where global == 0
        normalized_target = target_level - 1
        if normalized_target < len(drill):
            return drill[normalized_target:]
        else:
            level_name = {1: "Global", 2: "Super Region", 3: "Region", 4: "Country",
                          5: "Subnational 1"}.get(target_level, str(target_level))
            raise ValueError(
                f"Level '{level_name}' selected but current location is higher in the hierarchy than that")
