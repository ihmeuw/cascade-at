from cascade_at.core.db import db_queries


class Demographics:
    def __init__(self, gbd_round_id):
        """
        Demographic groups needed for shared functions.
        """
        dems = db_queries.get_demographics(gbd_team='epi', gbd_round_id=gbd_round_id)
        self.age_group_id = dems['age_group_id']
        self.location_id = dems['location_id']
        self.location_id = [101, 102]
        self.year_id = dems['year_id']
        self.sex_id = dems['sex_id']
