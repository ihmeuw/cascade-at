import pandas as pd

from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.inputs.demographics import Demographics

LOG = get_loggers(__name__)


class Population(BaseInput):
    def __init__(self, demographics: Demographics, decomp_step: str, gbd_round_id: int):
        """
        Gets population for all demographic groups. This is *not*
        and input for DisMod-AT (and therefore does not subclass
        BaseInput. It is just used to do covariate interpolation
        over non-standard age groups and years.

        Parameters
        ----------
        demographics
            A demographics object
        decomp_step
            The decomp step
        gbd_round_id
            The gbd round
        """
        super().__init__(gbd_round_id=gbd_round_id)
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id

        self.raw = None

    def get_population(self):
        """
        Gets the population counts from the database
        for the specified demographic group.
        """
        self.raw = db_queries.get_population(
            age_group_id=self.demographics.age_group_id,
            sex_id=self.demographics.sex_id,
            year_id=self.demographics.year_id,
            location_id=-1,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        )
        return self
    
    def configure_for_dismod(self) -> pd.DataFrame:
        """
        Configures population inputs for use in dismod
        by converting to age lower and upper from GBD age groups.
        """
        df = self.convert_to_age_lower_upper(self.raw)
        return df
