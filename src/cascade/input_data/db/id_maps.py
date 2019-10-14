from db_queries import get_ids


def map_variables_to_id(df, variables):
    """
    Maps a data frame with a variable specified in df
    and gets the associated table to get the ID.
    Args:
        df (pd.DataFrame):
        variables (list of str):

    Returns:
        A data frame with ID columns appended
        to the original df
    """
    for v in variables:
        id_map = get_ids(table=v)
        df = df.merge(id_map, on=v, how='left')
    return df


