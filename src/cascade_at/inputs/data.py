from typing import List, Optional

import pandas as pd

from cascade_at.core.db import elmo
from cascade_at.core.log import get_loggers
from cascade_at.dismod.integrand_mappings import make_integrand_map
from cascade_at.inputs.base_input import BaseInput
from cascade_at.inputs.demographics import Demographics
from cascade_at.inputs.uncertainty import stdev_from_crosswalk_version
from cascade_at.inputs.utilities import gbd_ids
from cascade_at.inputs.utilities.transformations import RELABEL_INCIDENCE_MAP

LOG = get_loggers(__name__)


class CrosswalkVersion(BaseInput):
    def __init__(self, crosswalk_version_id: int, exclude_outliers: bool,
                 demographics: Demographics, conn_def: str, gbd_round_id: int):
        """
        Pulls and formats all of the data from a crosswalk version in the epi database.

        Parameters
        ----------
        crosswalk_version_id
            The crosswalk version to pull from
        exclude_outliers
            whether to exclude outliers
        conn_def
            database connection definition
        gbd_round_id
            The GBD round
        demographics
            The demographics object
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
        """
        LOG.info(f"Getting crosswalk version for {self.crosswalk_version_id}.")
        import sys
        if 'darwin' in sys.platform:
            LOG.error(f"FIXME gma -- this call to elmo.get_crosswalk_version ought to contain an error_log_path argument.")
            LOG.error(f"FIXME gma -- START -- This call somehow switches logging from stdout to a socket.")
        self.raw = elmo.get_crosswalk_version(crosswalk_version_id=self.crosswalk_version_id)
        if 'darwin' in sys.platform:
            LOG.error(f"FIXME gma -- END --   Now logging to a socket. LOG.handlers: {LOG.handlers}")
        return self

    def configure_for_dismod(self, relabel_incidence: int,
                             measures_to_exclude: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Configures the crosswalk version for DisMod.

        Parameters
        ----------
        measures_to_exclude
            list of parameters to exclude, by name
        relabel_incidence
            how to label incidence -- see RELABEL_INCIDENCE_MAP
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

        df["age_lower"] = df["age_start"]
        df["time_lower"] = df["year_start"]
        df["age_upper"] = df["age_end"]
        df["time_upper"] = df["year_end"]

        df = self.get_out_of_demographic_notation(df, columns=['age', 'time'])

        df["meas_value"] = df["mean"]
        df["meas_std"] = stdev_from_crosswalk_version(df)
        df["name"] = df.seq.astype(str)

        df = self.keep_only_necessary_columns(df)

        return df

    @staticmethod
    def map_to_integrands(df: pd.DataFrame, relabel_incidence: int):
        """
        Maps the data from the IHME databases to the integrands expected by DisMod AT.

        Parameters
        ----------
        df
            A data frame to map to integrands
        relabel_incidence
            A relabel incidence code.
            Can be found in :py:class:`~cascade_at.inputs.utilities.transformations.RELABEL_INCIDENCE_MAP`
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


