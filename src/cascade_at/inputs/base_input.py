import pandas as pd
import numpy as np

from cascade_at.inputs.utilities.gbd_ids import get_age_group_metadata


class BaseInput:
    def __init__(self, gbd_round_id):
        self.age_group_metadata = get_age_group_metadata(gbd_round_id=gbd_round_id)
        self.columns_to_keep = [
            'location_id', 'time_lower', 'time_upper', 'sex_id',
            'measure', 'meas_value', 'meas_std',
            'age_lower', 'age_upper', 'age_group_id',
            'name', 'hold_out', 'density', 'eta', 'nu'
        ]

    def convert_to_age_lower_upper(self, df):
        """
        Converts a data frame that has age_group_id to
        age lower and upper based on age group metadata.
        :param df: (pd.DataFrame) data frame with column
            age_group_id
        :return: (pd.DataFrame)
        """
        df.age_group_id = df.age_group_id.astype(int)
        df = df.merge(self.age_group_metadata, on='age_group_id')
        return df

    def keep_only_necessary_columns(self, df):
        """
        Strips the data frame of unnecessary columns
        in prepping it for DisMod database.

        :param df: (pd.DataFrame)
        :return:
        """
        cols = df.columns
        keep_cols = [c for c in cols if c in self.columns_to_keep]
        return df[keep_cols]

    @staticmethod
    def get_out_of_demographic_notation(df, columns):
        """
        Convert things that are in demographic notation to non-demographic notation.
        """
        dd = df.copy()
        for col in columns:
            demographers = dd[col + '_lower'] == dd[col + '_upper']
            dd.loc[demographers, col + '_upper'] = dd.loc[demographers, col + '_lower'] + 1
        return dd    
