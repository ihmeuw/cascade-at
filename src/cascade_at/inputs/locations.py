import networkx as nx

from cascade_at.inputs.utilities.ids import CascadeConstants
from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class LocationDAG:
    def __init__(self, location_set_version_id):
        """
        Create a location DAG from the GBD location hierarchy, using
        networkx graph where each node is the location ID, and its properties
        are all properties from db_queries.

        The root of this dag is the global location ID.
        """
        LOG.info(f"Creating a location DAG for location_set_version_id {location_set_version_id}")
        self.location_set_version_id = location_set_version_id

        self.df = db_queries.get_location_metadata(
            location_set_version_id=location_set_version_id
        )

        dag = nx.DiGraph()
        dag.add_nodes_from([
            (int(row.location_id), row._asdict()) for row in self.df.itertuples()
        ])
        dag.add_edges_from([
            (int(row.parent_id), int(row.location_id))
            for row in self.df.loc[self.df.location_id != CascadeConstants.GLOBAL_LOCATION_ID].itertuples()
        ])
        dag.graph["root"] = CascadeConstants.GLOBAL_LOCATION_ID

