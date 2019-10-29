import numpy as np

import elmo

from cascade_at.dismod.integrand_mappings import make_integrand_map
from cascade_at.inputs.utilities import gbd_ids
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.inputs.uncertainty import stdev_from_crosswalk_version

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

        self.integrand_map = make_integrand_map()

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

    def configure_for_dismod(self,
                             data_eta, density, nu,
                             measures_to_exclude=None):
        """
        Configures the crosswalk version for DisMod.

        :param data_eta: (Dict[str, float]): Default value for eta parameter on distributions
            as a dictionary from measure name to float
        :param density: (Dict[str, float]): Default values for density parameter on distributions
            as a dictionary from measure name to string
        :param nu: (Dict[str, float]): The parameter for students-t distributions
        :param measures_to_exclude: (list) list of parameters to exclude, by name
        :return: pd.DataFrame
        """
        df = self.raw.copy()
        if self.exclude_outliers:
            df = df.loc[df.is_outlier != 1].copy()

        sex_ids = gbd_ids.get_sex_ids()
        measure_ids = gbd_ids.get_measure_ids(conn_def=self.conn_def)

        df = df.merge(sex_ids, on='sex')
        df = df.merge(measure_ids, on='measure')
        df = df.loc[~df.input_type.isin(['parent', 'group_review'])].copy()

        df = self.map_to_integrands(df)
        if measures_to_exclude:
            df['hold_out'] = 0
            df.loc[df.measure.isin(measures_to_exclude), 'hold_out'] = 1
            LOG.info(
                f"Filtering {df.hold_out.sum()} rows of of data where the measure has been excluded. "
                f"Measures marked for exclusion: {measures_to_exclude}. "
                f"{len(df)} rows remaining."
            )

        df["density"] = df.measure.apply(density.__getitem__)
        df["eta"] = df.measure.apply(data_eta.__getitem__)
        df["nu"] = df.measure.apply(nu.__getitem__)

        df.rename(columns={
            'age_start': 'age_lower',
            'age_end': 'age_upper'
        }, inplace=True)

        df["time_lower"] = df.year_start.astype(np.float)
        df["time_upper"] = df.year_end.astype(np.float)
        df["stdev"] = stdev_from_crosswalk_version(df)
        df["name"] = df.seq.astype(str)

        return df

    def map_to_integrands(self, df):
        """
        Maps the data from the IHME databases to the integrands expected by DisMod AT
        :param df:
        :return:
        """
        if any(df.measure_id == 6):
            LOG.warning(f"Found incidence, measure_id=6, in data. Should be Tincidence or Sincidence.")
        if any(df.measure_id == 17):
            LOG.info(
                f"Found case fatality rate, measure_id=17, in data. Ignoring it because it does not "
                f"map to a Dismod-AT integrand and cannot be used by the model."
            )
            df = df[df.measure_id != 17]

        try:
            df["measure"] = df.measure_id.apply(lambda k: self.integrand_map[k].name)
        except KeyError as ke:
            raise RuntimeError(
                f"The bundle data uses measure {str(ke)} which does not map "
                f"to an integrand. The map is {self.integrand_map}."
            )
        return df


