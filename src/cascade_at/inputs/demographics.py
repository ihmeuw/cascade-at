from typing import Optional

from cascade_at.core.db import db_queries
from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.utilites.gbd_ids import CascadeConstants


class Demographics:
    def __init__(self,
                 gbd_round_id: int,
                 location_set_version_id: Optional[int] = None):
        """
        Demographic groups needed for shared functions.
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
            self.mortality_rate_location_id = list(location_dag.dag.nodes)
        else:
            self.location_id = []
            self.mortality_rate_location_id = []
