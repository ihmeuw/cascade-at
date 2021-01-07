import pandas as pd
import numpy as np

from cascade_at.core.db import db_queries
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.dismod.constants import IntegrandEnum
from cascade_at.inputs.uncertainty import bounds_to_stdev
from cascade_at.inputs.utilities.gbd_ids import CascadeConstants
from cascade_at.inputs.demographics import Demographics

LOG = get_loggers(__name__)


class ASDR(BaseInput):
    def __init__(self, demographics: Demographics, decomp_step: str,
                 gbd_round_id: int):
        """
        Gets age-specific all-cause death rate for all
        demographic groups.

        Parameters
        ----------
        demographics
        decomp_step
        gbd_round_id
        """
        super().__init__(gbd_round_id=gbd_round_id)
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id

        self.raw = None

    def get_raw(self):
        """
        Pulls the raw ASDR and assigns them to this
        class.
        """
        LOG.info("Getting ASDR from get_envelope.")
        # location_ids = self.demographics.drill_locations
        location_ids = self.demographics.location_id
        LOG.info(f"Location_id's: {location_ids}")
        self.raw = db_queries.get_envelope(
            age_group_id=self.demographics.age_group_id,
            sex_id=self.demographics.sex_id,
            year_id=self.demographics.year_id,
            location_id=location_ids,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id,
            with_hiv=CascadeConstants.WITH_HIV,
            with_shock=CascadeConstants.WITH_SHOCK,
            rates=1
        )
        return self

    def configure_for_dismod(self, hold_out: int = 0) -> pd.DataFrame:
        """
        Configures ASDR for DisMod.

        Parameters
        ----------
        hold_out
            hold-out value for Dismod. 0 means it will be fit, 1 means held out
        """
        df = self.raw[[
            'age_group_id', 'location_id', 'year_id', 'sex_id', 'mean', 'upper', 'lower'
        ]].copy()
        df.rename(columns={
            'mean': 'meas_value',
            'year_id': 'time_lower'
        }, inplace=True)
        df['time_upper'] = df['time_lower'] + 1
        df = self.convert_to_age_lower_upper(df)
        df['integrand_id'] = IntegrandEnum.mtall.value
        df['measure'] = IntegrandEnum.mtall.name
        df['meas_std'] = bounds_to_stdev(lower=df.lower, upper=df.upper)

        df = self.keep_only_necessary_columns(df)
        df["hold_out"] = hold_out
        return df


