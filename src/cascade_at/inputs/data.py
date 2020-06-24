import numpy as np

from cascade_at.core.db import elmo
from cascade_at.dismod.integrand_mappings import make_integrand_map
from cascade_at.inputs.utilities.transformations import RELABEL_INCIDENCE_MAP
from cascade_at.inputs.utilities import gbd_ids
from cascade_at.core.log import get_loggers
from cascade_at.inputs.base_input import BaseInput
from cascade_at.inputs.uncertainty import stdev_from_crosswalk_version

LOG = get_loggers(__name__)


class CrosswalkVersion(BaseInput):
    def __init__(self, crosswalk_version_id, exclude_outliers,
                 demographics, conn_def, gbd_round_id):
        """
        :param crosswalk_version_id: (int)
        :param exclude_outliers: (bool) whether to exclude outliers
        :param conn_def: (str) connection definition
        """
        super().__init__(gbd_round_id=gbd_round_id)
        self.crosswalk_version_id = crosswalk_version_id
        self.exclude_outliers = exclude_outliers
        self.demographics = demographics
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

    def configure_for_dismod(self, relabel_incidence, measures_to_exclude=None):
        """
        Configures the crosswalk version for DisMod.

        :param measures_to_exclude: (list) list of parameters to exclude, by name
        :param relabel_incidence: (int) how to label incidence -- see RELABEL_INCIDENCE_MAP
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
        df = df.loc[df.location_id.isin(self.demographics.location_id)]
        df['hold_out'] = 0

        df = self.map_to_integrands(df, relabel_incidence=relabel_incidence)
        if measures_to_exclude:
            df.loc[df.measure.isin(measures_to_exclude), 'hold_out'] = 1
            LOG.info(
                f"Filtering {df.hold_out.sum()} rows of of data where the measure has been excluded. "
                f"Measures marked for exclusion: {measures_to_exclude}. "
                f"{len(df)} rows remaining."
            )

        df = df.loc[df.location_id.isin(self.demographics.location_id)]
        df = df.loc[df.sex_id.isin(self.demographics.sex_id)]

        df['age_lower'] = (df.age_start.astype(np.float) + df.age_end.astype(np.float)) / 2
        df['age_upper'] = df.age_lower

        df["time_lower"] = (df.year_start.astype(np.float) + df.year_end.astype(np.float)) / 2
        df["time_upper"] = df.time_lower
        df["meas_value"] = df["mean"]
        df["meas_std"] = stdev_from_crosswalk_version(df)
        df["name"] = df.seq.astype(str)

        df = self.get_out_of_demographic_notation(df, columns=['age', 'time'])
        df = self.keep_only_necessary_columns(df)

        return df

    @staticmethod
    def map_to_integrands(df, relabel_incidence):
        """
        Maps the data from the IHME databases to the integrands expected by DisMod AT
        :param df: (pd.DataFrame)
        :param relabel_incidence: (int)
        :return:
        """
        integrand_map = make_integrand_map()

        if any(df.measure_id == 17):
            LOG.info(
                f"Found case fatality rate, measure_id=17, in data. Ignoring it because it does not "
                f"map to a Dismod-AT integrand and cannot be used by the model."
            )
            df = df[df.measure_id != 17]

        try:
            df["measure"] = df.measure_id.apply(lambda k: integrand_map[k].name)
        except KeyError as ke:
            raise RuntimeError(
                f"The bundle data uses measure {str(ke)} which does not map "
                f"to an integrand. The map is {integrand_map}."
            )
        measure_dict = {measure: measure for measure in df.measure.unique().tolist()}
        measure_dict.update(RELABEL_INCIDENCE_MAP[relabel_incidence])
        df["measure"] = df["measure"].map(measure_dict)
        
        if any(df.measure == 'incidence'):
            LOG.error(f"Found incidence, measure_id=6, in data. Should be Tincidence or Sincidence.")
            raise ValueError("Measure ID cannot be 6 for incidence. Must be S or Tincidence.")
        
        return df


