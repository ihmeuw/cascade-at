from db_queries import get_outputs
import gbd.constants as gbd

from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput

LOG = get_loggers(__name__)


class CSMR(BaseInput):
    def __init__(self, process_version_id, cause_id, demographics,
                 decomp_step, gbd_round_id):
        """
        Get cause-specific mortality rate
        for demographic groups from a specific
        CodCorrect output version.

        :param process_version_id: (int)
        :param cause_id: (int)
        :param demographics (cascade_at.inputs.demographics.Demographics)
        :param decomp_step: (str)
        :param gbd_round_id: (int)
        """
        super().__init__()
        self.process_version_id = process_version_id
        self.cause_id = cause_id
        self.demographics = demographics
        self.decomp_step = decomp_step
        self.gbd_round_id = gbd_round_id

        self.raw = None

    def get_raw(self):
        """
        Pulls the raw CSMR and assigns it to
        this class.
        :return: self
        """
        LOG.info(f"Getting CSMR from process version ID {self.process_version_id}")
        self.raw = get_outputs(
            topic='cause',
            cause_id=self.cause_id,
            metric_id=gbd.metrics.RATE,
            measure_id=gbd.measures.DEATH,
            year_id=self.demographics.year_id,
            location_id=self.demographics.location_id,
            sex_id=self.demographics.sex_id,
            age_group_id=self.demographics.age_group_id,
            gbd_round_id=self.gbd_round_id,
            # TODO: these next two are hard-coded,
            #  should be self.decomp_step and self.process_version_id
            decomp_step='step4',
            process_version_id=14469
        )
        return self

    def configure_for_dismod(self):
        """
        Configures CSMR for DisMod.
        :return: (pd.DataFrame)
        """
        df = self.raw.rename(columns={
            "val": "meas_value",
            "year_id": "time_lower"
        })
        df["time_upper"] = df["time_lower"] + 1
        self.convert_to_age_lower_upper(df)
        return df

