from typing import Optional

from cascade_at.core.db import db_queries
from cascade_at.inputs.locations import LocationDAG


class Demographics:
    def __init__(self,
                 gbd_round_id: int,
                 location_set_version_id: Optional[int] = None):
        """
        Grabs and stores demographic information needed for shared functions.
        Will also make a location hierarchy dag.

        Parameters
        ----------
        gbd_round_id
            The GBD round
        location_set_version_id
            The location set version to use (right now EpiViz-AT is passing
            dismod location set versions, but this will eventually switch
            to the cause of death hierarchy that is more extensive).
        """
        demographics = db_queries.get_demographics(
            gbd_team='epi', gbd_round_id=gbd_round_id)
        self.age_group_id = demographics['age_group_id']
        self.sex_id = demographics['sex_id'] + [3]

        cod_demographics = db_queries.get_demographics(
            gbd_team='cod', gbd_round_id=gbd_round_id)
        self.year_id = cod_demographics['year_id']

        if location_set_version_id:
            location_dag = LocationDAG(
                location_set_version_id=location_set_version_id,
                gbd_round_id=gbd_round_id)
            self.location_id = list(location_dag.dag.nodes)
            self.drill_locations = list(location_dag.dag.nodes)
        else:
            self.location_id = []
            self.drill_locations = []
