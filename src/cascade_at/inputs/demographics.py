from cascade_at.core.db import db_queries


class Demographics:
    def __init__(self, gbd_round_id):
        """
        Demographic groups needed for shared functions.
        """
        demographics = db_queries.get_demographics(gbd_team='epi', gbd_round_id=gbd_round_id)
        self.age_group_id = demographics['age_group_id']
        self.location_id = demographics['location_id']
        self.sex_id = demographics['sex_id'] + [3]
        
        cod_demographics = db_queries.get_demographics(gbd_team='cod', gbd_round_id=gbd_round_id)
        self.year_id = cod_demographics['year_id']
