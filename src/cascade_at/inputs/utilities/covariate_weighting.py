from functools import lru_cache

from cascade_at.core.log import get_loggers
from cascade_at.inputs.utilities.gbd_ids import get_age_id_to_range, get_sex_ids

LOG = get_loggers(__name__)

AGE_ID_TO_RANGE = get_age_id_to_range()
SEX_IDS = get_sex_ids()['sex_id'].tolist()


def get_age_year_value(cov, loc_ids, sex_ids):
    """
    Build a dictionary where key is (loc_id, age_id) and value is
    a list of (age_group_id, year_id, mean_value) collected from
    covariate data. If covariate from a certain location is not
    available, the list is empty.

    Args:
        cov (pandas.Dataframe): a dataframe storing covariate info
        loc_ids (list[int]): location ids
        sex_ids (list[int]): sex ids

    Returns:
        dct (dict[tuple(int, int), list[tuple(int, int, float)]])

    """
    dct = {}
    available_locations = set(cov.location_id.unique())
    for loc_id in loc_ids:
        for sex_id in sex_ids:
            if loc_id in available_locations:
                cov_sub = cov[(cov['location_id'] == loc_id) & (cov['sex_id'].isin([3, sex_id]))]
                cov_sub = cov_sub.sort_values(['age_group_id', 'year_id'])
                dct[(loc_id, sex_id)] = list(cov_sub[['age_group_id', 'year_id', 'mean_value']].values)
            else:
                dct[(loc_id, sex_id)] = []
    return dct


def intersect(age_start, age_end, year_start, year_end, tuples):
    """
    Find covariate entries that intersects with a given measurement
    entry and compute weights based on length of the overlap.

    Args:
        age_start (int): start age of the measurement entry
        age_end (int): end age of the measurement entry
        year_start (int): start year of the measurement entry
        year_end (int): end year of the entry
        tuples (tuple(int, int, float)): tuples of (age_group_id, year_id, cov_value)
    Returns:
        common_tuples (list[tuple(int, int, float)]): tuples that intersect with measurement
        weights (list[float]): weights corresponding to each tuple
    """
    common_tuples = []
    weights = []
    for tup in tuples:
        tup = (int(tup[0]), int(tup[1]), tup[2])
        age_group = tup[0]
        if age_group in AGE_ID_TO_RANGE:
            year = tup[1]
            interval = AGE_ID_TO_RANGE[age_group]
            if year_start <= year <= year_end:  # check if intersects in time
                if interval[0] < age_end and interval[1] > age_start:  # check if intersect in age
                    common_tuples.append(tup)
                    weights.append(max(min(age_end+1, interval[1]) - max(age_start, interval[0]), 0) /
                                   (interval[1] - interval[0]))
                elif age_start == age_end and \
                        (interval[0] == age_start or interval[1] == age_end):  # case when measurement is on boundary
                    common_tuples.append(tup)
                    weights.append(1./(interval[1] - interval[0]))
    return common_tuples, weights


def pop_val_dict(df, locations):
    """
    Build a dictionary mapping (location_id, sex_id, age_group_id, year_id) to
    a population value.
    Args:
        df (pandas.Dataframe): population data
        locations (list[int]): location ids

    Returns:
        dct (dict[tuple(int, int, int, int), float])
    """
    dct = {}
    for i, row in df[df['location_id'].isin(locations)].iterrows():
        dct[(row['location_id'], row['sex_id'], row['age_group_id'], row['year_id'])] = row['population']
    return dct


def get_interpolated_covariate_values(data_df, covariate_df, population_df):
    """
    Gets the unique age-time combinations from the data_df, and creates
    interpolated covariate values for each of these combinations by population-weighting
    the standard GBD age-years that span the non-standard combinations.

    :param data_df: (pd.DataFrame)
    :param covariate_df: (pd.DataFrame)
    :param population_df: (pd.DataFrame)
    :return:
    """
    meas = data_df.copy()

    loc_ids = sorted(meas.location_id.unique())
    cov_age_year_value = get_age_year_value(covariate_df, loc_ids, SEX_IDS)
    pop_dict = pop_val_dict(population_df, loc_ids)
    
    meas['mean_value'] = 0.0

    for i, row in meas.iterrows():
        if (i + 1) % 500 == 0:
            print('processed', i + 1, 'rows', end='\r')
        dct = {}
        tuples, weights = intersect(
            age_start=row['age_lower'],
            age_end=row['age_upper'],
            year_start=row['time_lower'],
            year_end=row['time_upper'],
            tuples=cov_age_year_value[(row['location_id'], row['sex_id'])]
        )
        dct['val'] = [tup[2] for tup in tuples]  # list of covariate values
        dct['wts'] = weights
        dct['pop'] = []  # to store list of population values corresponding to tuples
        val = 0.0
        total_wts = 0.0
        if len(tuples) > 0:
            for j in range(len(tuples)):
                if row['sex_id'] != 3:
                    dct['pop'].append(
                        pop_dict[(row['location_id'], row['sex_id'], tuples[j][0], tuples[j][1])]
                    )
                else:
                    dct['pop'].append(
                        pop_dict[(row['location_id'], 1, tuples[j][0], tuples[j][1])] +
                        pop_dict[(row['location_id'], 2, tuples[j][0], tuples[j][1])]
                    )
                val += tuples[j][2] * weights[j] * dct['pop'][-1]  # weigh covariate value by population
                total_wts += weights[j] * dct['pop'][-1]
            val /= total_wts
        meas.loc[i, 'mean_value'] = val

    return meas['mean_value']
