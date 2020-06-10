import networkx as nx
import numpy as np
import pandas as pd
from typing import List

from cascade_at.inputs.utilities.gbd_ids import CascadeConstants
from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class LocationDAG:
    def __init__(self, location_set_version_id, gbd_round_id):
        """
        Create a location DAG from the GBD location hierarchy, using
        networkx graph where each node is the location ID, and its properties
        are all properties from db_queries.

        The root of this dag is the global location ID.
        """
        LOG.info(
            f"Creating a location DAG for location_set_version_id "
            f"{location_set_version_id}")
        self.location_set_version_id = location_set_version_id

        self.df = db_queries.get_location_metadata(
            location_set_version_id=location_set_version_id,
            location_set_id=CascadeConstants.ESTIMATION_LOCATION_HIERARCHY_ID,
            gbd_round_id=gbd_round_id
        )

        self.dag = nx.DiGraph()
        for index, row in self.df.iterrows():
            self.dag.add_node(int(row['location_id']), **row.to_dict())
        self.dag.add_edges_from([
            (int(row.parent_id), int(row.location_id))
            for row in self.df.loc[
                self.df.location_id != CascadeConstants.GLOBAL_LOCATION_ID].itertuples()
        ])
        self.dag.graph["root"] = CascadeConstants.GLOBAL_LOCATION_ID

    def descendants(self, location_id):
        """
        Gets all descendants (not just direct children) for a location ID.
        :param location_id: (int)
        :return:
        """
        return nx.algorithms.dag.descendants(G=self.dag, source=location_id)

    def parent_children(self, location_id):
        """
        Gets the parent and the child location IDs.
        :param location_id: (int)
        :return:
        """
        return [location_id] + list(self.dag.successors(location_id))

    def to_dataframe(self):
        """
        Converts the location DAG to a data frame with location ID and parent
        ID and name. Helpful for debugging, and putting into the dismod
        database.

        Returns:
            pd.DataFrame
        """
        sorted_locations = list(nx.lexicographical_topological_sort(self.dag))
        parents = list()
        names = list()
        for loc in sorted_locations:
            parent = list(self.dag.predecessors(loc))
            if parent:
                parents.append(int(parent[0]))
            else:
                parents.append(np.nan)
            names.append(self.dag.nodes[loc]["location_name"])
        return pd.DataFrame(dict(
            location_id=sorted_locations,
            parent_id=parents,
            name=names
        ))


def locations_by_drill(drill_location_start: int, drill_location_end: List[int], dag: LocationDAG):
    if not drill_location_start and drill_location_end:
        raise ValueError(
            "A location_drill_start must be specified in order "
            "to perform a location drill.")

    elif drill_location_start and not drill_location_end:
        LOG.info(
            f"This is a DRILL model, so only going to pull data "
            f"associated with drill location start "
            f"{drill_location_start} and its descendants."
        )
        drill_locations_all = ([drill_location_start] + list(dag.descendants(
                                location_id=drill_location_start)))
        drill_locations = list(
            dag.parent_children(drill_location_start))
    elif drill_location_start and drill_location_end:
        LOG.info(
            f"This is a DRILL model, so only data for "
            f"{drill_location_start} (the parent) and descendents "
            f"of {drill_location_end} (the children) will be pulled."
        )
        drill_locations_all = [drill_location_start]
        for child in drill_location_end:
            drill_locations_all.append(child)
            drill_locations_all = drill_locations_all + list(
                dag.descendants(location_id=child))
        drill_locations = [drill_location_start] + drill_location_end
    else:
        drill_locations_all = None
        drill_locations = None
    return drill_locations_all, drill_locations
