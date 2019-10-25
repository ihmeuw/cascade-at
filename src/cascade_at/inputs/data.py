import elmo

from cascade_at.inputs.utilities import ids
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput

LOG = get_loggers(__name__)


class CrosswalkVersion(BaseInput):
    def __init__(self, crosswalk_version_id, exclude_outliers, conn_def):
        """
        :param crosswalk_version_id: (int)
        :param exclude_outliers: (bool) whether to exclude outliers
        :param conn_def: (str) connection definition
        """
        super().__init__()
        self.crosswalk_version_id = crosswalk_version_id
        self.exclude_outliers = exclude_outliers
        self.conn_def = conn_def

        self.raw = None

    def get_raw(self):
        """
        Pulls the raw crosswalk version from the database.
        These are the observations that will be used in the bundle.
        :return: self
        """
        LOG.info(f"Getting crosswalk version for {self.crosswalk_version_id}.")
        self.raw = elmo.get_crosswalk_version(crosswalk_version_id=self.crosswalk_version_id)
        return self

    def configure_for_dismod(self):
        """
        Configures the crosswalk version for DisMod.
        :return: pd.DataFrame
        """
        df = self.raw.copy()
        if self.exclude_outliers:
            df = df.loc[df.is_outlier != 1].copy()

        sex_ids = ids.get_sex_ids()
        measure_ids = ids.get_measure_ids(conn_def=self.conn_def)

        df = df.merge(sex_ids, on='sex')
        df = df.merge(measure_ids, on='measure')

        df = df.loc[~df.input_type.isin(['parent', 'group_review'])].copy()
        df = self.convert_to_age_lower_upper(df)
        return df
