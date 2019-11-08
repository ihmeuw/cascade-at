from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class Population:
    def __init__(self, demographics, decomp_step, gbd_round_id):
        """
        Gets population for all demographic groups. This is *not*
        and input for DisMod-AT (and therefore does not subclass
        BaseInput. It is just used to do covariate interpolation
        over non-standard age groups and years.

        :param demographics: (cascade_at.inputs.demographics.Demographics)
        :param decomp_step: (int)
        :param gbd_round_id: (int)
        """
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id

        self.raw = None

    def get_population(self):
        """
        Gets the population counts from the database
        for the specified demographic group.

        :return:
        """
        self.raw = db_queries.get_population(
            age_group_id=self.demographics.age_group_id,
            sex_id=self.demographics.sex_id,
            year_id=self.demographics.year_id,
            location_id=self.demographics.location_id,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        )
        return self
