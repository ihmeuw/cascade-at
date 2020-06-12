import pandas as pd
import numpy as np
import os
from typing import List, Optional, Dict

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.integrand_mappings import reverse_integrand_map
from cascade_at.dismod.integrand_mappings import PRIMARY_INTEGRANDS_TO_RATES
from cascade_at.dismod.api import DismodAPIError

LOG = get_loggers(__name__)


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

    def get_predictions(self, location_id: Optional[int] = None, sex_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get the predictions from the predict table for a specific
        location (rather than node) and sex ID.
        """
        predictions = self.predict.merge(self.avgint, on=['avgint_id'])
        predictions = predictions.merge(self.integrand, on=['integrand_id'])
        predictions['rate'] = predictions['integrand_name'].map(PRIMARY_INTEGRANDS_TO_RATES)
        if location_id is not None:
            predictions = predictions.loc[predictions.c_location_id == location_id].copy()
        if sex_id is not None:
            predictions = predictions.loc[predictions.c_sex_id == sex_id].copy()
        return predictions

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

        df = self.get_predictions(location_id=location_id, sex_id=sex_id)

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
                    ]['avg_integrand'].values

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

    def format_predictions_for_ihme(self) -> pd.DataFrame:
        """
        Gets the predictions from the predict table and transforms them
        into the GBD ids that we expect.
        :return:
        """
        predictions = self.get_predictions()[[
            'c_location_id', 'c_age_group_id', 'c_year_id', 'c_sex_id',
            'integrand_name', 'time_lower', 'time_upper', 'age_lower', 'age_upper',
            'avg_integrand'
        ]]
        gbd_id_cols = ['location_id', 'sex_id', 'age_group_id', 'year_id']

        predictions.rename(columns={'c_' + x: x for x in gbd_id_cols}, inplace=True)
        for col in gbd_id_cols:
            predictions[col] = predictions[col].astype(int)
        predictions.rename(columns={'avg_integrand': 'mean'}, inplace=True)
        predictions['lower'] = predictions['mean']
        predictions['upper'] = predictions['mean']

        integrand_map = reverse_integrand_map()
        predictions['measure_id'] = predictions.integrand_name.apply(lambda x: integrand_map[x])

        # Duplicate the Sincidence results to incidence hazard for the Viz tool
        predictions_2 = predictions.loc[predictions.measure_id == 41].copy()
        predictions_2['measure_id'] = 6
        predictions = pd.concat([predictions, predictions_2], axis=0)
        
        return predictions[[
            'location_id', 'age_group_id', 'year_id', 'sex_id',
            'measure_id', 'mean', 'upper', 'lower'
        ]]
