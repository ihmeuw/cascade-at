import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api import DismodAPIError
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.integrand_mappings import PRIMARY_INTEGRANDS_TO_RATES, reverse_integrand_map
from cascade_at.inputs.utilities.gbd_ids import DEMOGRAPHIC_ID_COLS
from cascade_at.inputs.utilities.gbd_ids import make_age_intervals, make_time_intervals
from cascade_at.inputs.utilities.gbd_ids import map_id_from_interval_tree

LOG = get_loggers(__name__)


class ExtractorCols:
    REQUIRED_DEMOGRAPHIC_COLS = ['location_id', 'sex_id']
    RESULT_COL = 'avg_integrand'
    SAMPLE_COL = 'sample_index'
    VALUE_COL_SAMPLES = 'draw'
    VALUE_COL_FIT = 'mean'


class DismodExtractorError(DismodAPIError):
    """Errors raised when there are issues with DismodExtractor."""
    pass


class DismodExtractor(DismodIO):
    """
    Sits on top of the DismodIO class,
    and takes everything from the collector module
    and puts them into the Dismod database tables
    in the correct construction.
    """
    def __init__(self, path):
        super().__init__(path=path)
        if not os.path.isfile(path):
            raise DismodExtractorError(f"SQLite file {str(path)} has not been created or filled yet.")

    def _extract_raw_predictions(self) -> pd.DataFrame:
        """
        Grab raw predictions from the predict table.
        """
        df = self.predict.merge(self.avgint, on=['avgint_id'])
        df = df.merge(self.integrand, on=['integrand_id'])
        df['rate'] = df['integrand_name'].map(
            PRIMARY_INTEGRANDS_TO_RATES
        )
        return df

    def get_predictions(self, locations: Optional[List[int]] = None,
                        sexes: Optional[List[int]] = None,
                        samples: bool = False) -> pd.DataFrame:
        """
        Get the predictions from the predict table for locations and sexes.
        Will either return a column of 'mean' if not samples, otherwise 'draw', which can then
        be reshaped wide if necessary.
        """
        df = self._extract_raw_predictions()
        if locations is not None:
            df = df.loc[df.c_location_id.isin(locations)].copy()
            if set(df.c_location_id.values) != set(locations):
                missing_locations = set(df.c_location_id.values) - set(locations)
                raise DismodExtractorError("The following locations you asked for were missing: "
                                           f"{missing_locations}.")
        if sexes is not None:
            df = df.loc[df.c_sex_id.isin(sexes)].copy()
            if set(df.c_sex_id.values) != set(sexes):
                missing_sexes = set(df.c_sex_id.values) - set(sexes)
                raise DismodExtractorError(f"The following sexes you asked for were missing: {missing_sexes}.")

        df.rename(
            columns={'c_' + x: x for x in DEMOGRAPHIC_ID_COLS}, inplace=True
        )
        DEMOGRAPHIC_COLS = []
        for col in ExtractorCols.REQUIRED_DEMOGRAPHIC_COLS:
            if col not in df.columns:
                raise DismodExtractorError(f"Cannot find required col {col} in the"
                                           "predictions columns: {predictions.columns}.")
            else:
                DEMOGRAPHIC_COLS.append(col)

        if samples:
            VALUE_COL = ExtractorCols.VALUE_COL_SAMPLES
            if ExtractorCols.SAMPLE_COL not in df.columns:
                raise DismodExtractorError("Cannot find sample index column. Are you sure you created samples?")
            if np.isnan(df[ExtractorCols.SAMPLE_COL]).all():
                raise DismodExtractorError("All sample index values are null. Are you sure you created samples?")
        else:
            VALUE_COL = ExtractorCols.VALUE_COL_FIT

        df.rename(columns={ExtractorCols.RESULT_COL: VALUE_COL}, inplace=True)
        return df[DEMOGRAPHIC_COLS + [
            'integrand_id', 'integrand_name', 'rate',
            'time_lower', 'time_upper', 'age_lower', 'age_upper', VALUE_COL
        ]]

    def gather_draws_for_prior_grid(self,
                                    location_id: int,
                                    sex_id: int,
                                    rates: List[str],
                                    value: bool = True,
                                    dage: bool = True,
                                    dtime: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Takes draws and formats them for a prior grid for values, dage, and dtime.
        Assumes that age_lower == age_upper and time_lower == time_upper for all
        data rows. We might not want to do all value, dage, and dtime, so pass False
        if you want to skip those.

        Arguments
        ---------
        location_id
        sex_id
        rates
            list of rates to get the draws for
        value
            whether to calculate value priors
        dage
            whether to calculate dage priors
        dtime
            whether to calculate dtime priors

        Returns
        -------
        Dictionary of 3-d arrays of value, dage, and dtime draws over age and time for this loc and sex
        """
        rate_dict = dict()
        for r in rates:
            rate_dict[r] = dict()

        df = self.get_predictions(locations=[location_id], sexes=[sex_id], samples=True)

        assert (df.age_lower.values == df.age_upper.values).all()
        assert (df.time_lower.values == df.time_upper.values).all()

        # Loop over rates, age, and time
        for r in rates:
            df2 = df.loc[df.rate == r].copy()

            ages = np.asarray(sorted(df2.age_lower.unique().tolist()))
            times = np.asarray(sorted(df2.time_lower.unique().tolist()))
            n_draws = int(len(df2) / (len(ages) * len(times)))

            # Save these for later for quality checks
            rate_dict[r]['ages'] = ages
            rate_dict[r]['times'] = times
            rate_dict[r]['n_draws'] = n_draws

            # Create template for filling in the draws
            draw_data = np.zeros((len(ages), len(times), n_draws))
            for age_idx, age in enumerate(ages):
                for time_idx, time in enumerate(times):
                    # Subset to the draws that we want from avg_integrand
                    # but only for this particular age and time
                    draws = df2.loc[
                        (df2.age_lower == age) &
                        (df2.time_lower == time)
                    ][ExtractorCols.VALUE_COL_SAMPLES].values

                    # Check to makes sure that the number of draws corresponds to the number
                    # of draws for the whole thing per age and time
                    assert len(draws) == n_draws
                    draw_data[age_idx, time_idx, :] = draws

            if value:
                rate_dict[r]['value'] = draw_data
            if dage:
                rate_dict[r]['dage'] = np.diff(draw_data, n=1, axis=0)
            if dtime:
                rate_dict[r]['dtime'] = np.diff(draw_data, n=1, axis=1)

        return rate_dict

    def format_predictions_for_ihme(self, gbd_round_id: int,
                                    locations: Optional[List[int]] = None,
                                    sexes: Optional[List[int]] = None,
                                    samples: bool = False) -> pd.DataFrame:
        """
        Formats predictions from the prediction table and returns either the mean
        or draws, based on whether or not samples is False or True.

        Parameters
        ----------
        locations
            A list of locations to extract from the predictions
        sexes
            A list of sexes to extract from the predictions
        gbd_round_id
            The GBD round ID to format the predictions for
        samples
            Whether or not the predictions have draws (samples) or whether
            it is just one fit.

        Returns
        -------
        Data frame with predictions formatted for the IHME databases.
        """
        pred = self.get_predictions(locations=locations, sexes=sexes, samples=samples)
        map_age = 'age_group_id' not in pred.columns
        map_year = 'year_id' not in pred.columns

        if map_age:
            age_intervals = make_age_intervals(gbd_round_id=gbd_round_id)
            pred['age_group_id'] = pred['age_lower'].apply(
                lambda x: map_id_from_interval_tree(index=x, tree=age_intervals)
            )
        if map_year:
            time_intervals = make_time_intervals()
            pred['year_id'] = pred['time_lower'].apply(
                lambda x: map_id_from_interval_tree(index=x, tree=time_intervals)
            )

        integrand_map = reverse_integrand_map()
        pred['measure_id'] = pred.integrand_name.apply(lambda x: integrand_map[x])

        # Duplicates the Sincidence results, if they exist, so that they
        # show up as incidence in the visualization tool
        if integrand_map['Sincidence'] in pred.integrand_name.unique():
            incidence = pred.loc[pred.measure_id == integrand_map['Sincidence']].copy()
            incidence['measure_id'] = integrand_map['incidence']
            pred = pd.concat([pred, incidence], axis=0).reset_index()

        return pred
