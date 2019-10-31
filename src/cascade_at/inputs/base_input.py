from cascade_at.inputs.utilities.gbd_ids import get_age_group_metadata


class BaseInput:
    def __init__(self):
        self.age_group_metadata = get_age_group_metadata()
        self.columns_to_keep = [
            'location_id', 'time_lower', 'time_upper', 'sex_id',
            'measure', 'meas_value', 'stdev',
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
