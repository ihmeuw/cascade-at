import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Optional

from cascade_at.inputs.utilities.gbd_ids import CascadeConstants
from cascade_at.core.db import db_queries
from cascade_at.core import CascadeATError
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class LocationDAGError(CascadeATError):
    pass


class LocationDAG:
    def __init__(self, location_set_version_id: Optional[int] = None,
                 gbd_round_id: Optional[int] = None, df: Optional[pd.DataFrame] = None,
                 root: Optional[int] = None):
        """
        Create a location DAG from the GBD location hierarchy, using
        networkx graph where each node is the location ID, and its properties
        are all properties from db_queries.

        The root of this dag is the global location ID.

        Parameters
        ----------
        location_set_version_id
            The location set version corresponding to the hierarchy to pull from the IHME
            databases
        gbd_round_id
            Which gbd round the location set version is coming from
        df
            An optional df to pass instead of location sets and gbd rounds if
            you'd rather construct the DAG from a pandas data frame.
        """
        if df is None:
            if location_set_version_id is None:
                raise LocationDAGError("Need to either pass a location set version ID or a data frame.")
            else:
                if gbd_round_id is None:
                    raise LocationDAGError("You must pass a GBD round ID with a location set"
                                           "version ID to the db_queries function.")
                LOG.info(
                    f"Creating a location DAG for location_set_version_id "
                    f"{location_set_version_id}")
                self.location_set_version_id = location_set_version_id

                self.df = db_queries.get_location_metadata(
                    location_set_version_id=location_set_version_id,
                    location_set_id=CascadeConstants.ESTIMATION_LOCATION_HIERARCHY_ID,
                    gbd_round_id=gbd_round_id
                )
                root = CascadeConstants.GLOBAL_LOCATION_ID
        else:
            if location_set_version_id is not None:
                LOG.warn("You passed both a location set version ID and a dataframe."
                         "Ignoring the location set version ID and creating the DAG based"
                         "on the data frame.")
            if root is None:
                raise LocationDAGError("Must specify a root if you're passing in a data frame.")
            self.df = df

        self.dag = nx.DiGraph()
        for index, row in self.df.iterrows():
            self.dag.add_node(int(row['location_id']), **row.to_dict())
        self.dag.add_edges_from([
            (int(row.parent_id), int(row.location_id))
            for row in self.df.loc[
                self.df.location_id != root].itertuples()
        ])
        self.dag.graph["root"] = root

    def depth(self, location_id: int) -> int:
        """
        Gets the depth of the hierarchy at this location.
        """
        return nx.shortest_path_length(G=self.dag, source=self.dag.graph["root"], target=location_id)

    def descendants(self, location_id: int) -> List[int]:
        """
        Gets all descendants (not just direct children) for a location ID.
        :param location_id: (int)
        :return:
        """
        return nx.algorithms.dag.descendants(G=self.dag, source=location_id)

    def children(self, location_id: int) -> List[int]:
        """
        Gets the child location IDs.
        """
        return list(self.dag.successors(location_id))

    def parent_children(self, location_id: int) -> List[int]:
        """
        Gets the parent and the child location IDs.
        """
        return [location_id] + list(self.dag.successors(location_id))

    def is_leaf(self, location_id: int) -> bool:
        """
        Checks if a location is a leaf node in the tree.
        """
        return len(list(self.dag.successors(location_id))) == 0

    def to_dataframe(self) -> pd.DataFrame:
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
