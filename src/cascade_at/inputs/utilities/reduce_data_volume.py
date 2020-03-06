import numpy as np


def decimate_years(data, num_years=5):
    """
    Reduce the volume of CSMR to only every num_years years. Requires
    that annual years are present in the data frame, and that it's square.
    ASSUMES that the name of the time columns are 'time_lower' and 'time_upper'
    where 'time_lower' was the original year ID from the GBD database.

    Args:
        data: (pd.DataFrame with columns 'time_lower' and 'time_upper', and 'meas_value' and 'meas_std'
        num_years: (int)

    Returns:
        (pd.DataFrame) with mid-pointed CSMR over num_years
    """
    df = data.copy()
    min_year = int(df.time_lower.min())
    max_year = int(df.time_lower.max())

    year_ids = list(range(min_year, max_year+1, 1))
    collapse_ids = np.repeat(list(range(min_year, max_year+1, num_years)), repeats=num_years)[:len(year_ids)]

    collapse_dict_start = {
        year_ids[i]: collapse_ids[i] for i in range(len(year_ids))
    }

    group_columns = [x for x in df.columns if x not in ['meas_value', 'meas_std']]

    df['time_lower'] = df['time_lower'].astype(int).map(collapse_dict_start) + num_years / 2
    df['time_upper'] = df['time_lower']

    group = df.groupby(group_columns).mean()
    group.reset_index(inplace=True)

    return group[group_columns + ['meas_value', 'meas_std']]
