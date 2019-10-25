from cascade_at.inputs.utilities.ids import get_age_group_metadata


class BaseInput:
    def __init__(self):
        self.age_group_metadata = get_age_group_metadata()

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

