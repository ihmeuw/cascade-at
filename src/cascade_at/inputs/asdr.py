
from db_queries import get_envelope

from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput

LOG = get_loggers(__name__)


class ASDR(BaseInput):
    def __init__(self, demographics, decomp_step,
                 gbd_round_id, with_hiv=True):
        """
        Gets age-specific death rate for all
        demographic groups.

        :param demographics: (cascade_at.inputs.demographics.Demographics)
        :param decomp_step: (int)
        :param gbd_round_id: (int)
        :param with_hiv: (bool) pull HIV-added envelope?
        :return:
        """
        super().__init__()
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id
        self.with_hiv = with_hiv

        self.raw = None

    def get_raw(self):
        """
        Pulls the raw ASDR and assigns them to this
        class.
        :return: self
        """
        LOG.info("Getting ASDR from get_envelope.")
        self.raw = get_envelope(
            age_group_id=self.demographics.age_group_id,
            sex_id=self.demographics.sex_id,
            year_id=self.demographics.year_id,
            location_id=self.demographics.location_id,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id,
            with_hiv=self.with_hiv
        )
        return self

    def configure_for_dismod(self):
        """
        Configures ASDR for DisMod.
        :return: (pd.DataFrame)
        """
        df = self.raw[[
            'age_group_id', 'location_id', 'year_id', 'sex_id', 'mean', 'upper', 'lower'
        ]]
        df.rename(columns={
            'mean': 'meas_value',
            'year_id': 'time_lower'
        }, inplace=True)
        df['time_upper'] = df['time_lower'] + 1
        df = self.convert_to_age_lower_upper(df)
        return df


