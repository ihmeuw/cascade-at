import pandas as pd

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.dismod_io import DismodIO
from cascade_at.dismod.integrand_mappings import reverse_integrand_map
from cascade_at.dismod.api.fill_extract_helpers import utils

LOG = get_loggers(__name__)


class DismodExtractor(DismodIO):
    """
    Sits on top of the DismodIO class,
    and takes everything from the collector module
    and puts them into the Dismod database tables
    in the correct construction.
    """
    def __init__(self, path):
        super().__init__(path=path)

    def format_predictions_for_ihme(self):
        """
        Gets the predictions from the predict table and transforms them
        into the GBD ids that we expect.
        :return:
        """
        predictions = self.predict.merge(self.avgint, on=['avgint_id'])
        predictions = predictions.merge(self.integrand, on=['integrand_id'])
        predictions = predictions[[
            'c_location_id', 'c_age_group_id', 'c_year_id', 'c_sex_id',
            'integrand_name',
            'time_lower', 'time_upper', 'age_lower', 'age_upper',
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

