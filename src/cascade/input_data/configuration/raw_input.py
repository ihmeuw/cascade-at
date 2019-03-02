import numpy as np
from numpy import dtype
import pandas as pd


EXPECTED_TYPES = dict(
    age_specific_death_rate=[
        ('location', dtype('int64')),
        ('time_lower', dtype('int64')),
        ('mean', dtype('float64')),
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_upper', dtype('int64')),
        ('std', dtype('float64')),
        ('integrand', dtype('O')),
        ('hold_out', dtype('int64')),
        ('density', dtype('O')),
        ('eta', dtype('float64')),
        ('nu', dtype('float64')),
    ],
    ages_df=[
        ('age_group_id', dtype('int64')),
        ('age_group_years_start', dtype('float64')),
        ('age_group_years_end', dtype('float64')),
        ('age_group_weight_value', dtype('float64')),
    ],
    average_integrand_cases=[
        ('integrand_name', dtype('O')),
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('int64')),
        ('time_upper', dtype('int64')),
        ('weight_id', dtype('int64')),
        ('node_id', dtype('int64')),
        ('sex_id', dtype('int64')),
    ],
    bundle=[
        ('seq', dtype('int64')),
        ('measure', dtype('O')),
        ('mean', dtype('float64')),
        ('sex', dtype('O')),
        ('sex_id', dtype('int64')),
        ('standard_error', dtype('float64')),
        ('hold_out', dtype('int64')),
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('int64')),
        ('time_upper', dtype('int64')),
        ('location_id', dtype('int64')),
    ],
    sparse_covariate_data=[
        ('study_covariate_id', dtype('int64')),
        ('seq', dtype('int64')),
        ('bundle_id', dtype('int64')),
    ],
    country_covariates=[
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('float64')),
        ('time_upper', dtype('float64')),
        ('sex_id', dtype('int64')),
        ('mean_value', dtype('float64')),
    ]
)
"""These are the data frames and column data types that define
the raw input data. If you want to mock input data or get it
from another source, it has to have these columns."""


def validate_input_data_types(input_data):
    """This is a gating function that insists the input data have only certain
    data sets with certain columns and data types."""
    not_matching = list()
    for member_name in dir(input_data):
        dtypes = dtype_of_member(input_data, member_name)
        if dtypes is not None:
            not_matching.extend(check_one_member(dtypes, member_name))
    return not_matching


def dtype_of_member(input_data, member_name):
    member = getattr(input_data, member_name)
    if isinstance(member, pd.DataFrame):
        dtypes = member.dtypes
    elif isinstance(member, dict) and member:
        # Look at the first dictionary in a dictionary of dataframes. Assume
        # the rest are the same.
        dict_member = next(iter(member.values()))
        if isinstance(dict_member, pd.DataFrame):
            dtypes = dict_member.dtypes
        else:
            dtypes = None
    else:
        dtypes = None
    return dtypes


def check_one_member(dtypes, member_name):
    not_matching = list()
    variable_dype = dict(zip(dtypes.index.tolist(), dtypes.tolist()))
    if member_name not in EXPECTED_TYPES:
        print(f"{member_name} {list(zip(dtypes.index.tolist(), dtypes.tolist()))}")
        not_matching.append((member_name, "all", dtype("O"), dtype("O")))
    else:
        expected = dict(EXPECTED_TYPES[member_name])
        for column, col_type in variable_dype.items():
            if column not in expected:
                not_matching.append((member_name, column, col_type, None))
            if not np.issubdtype(col_type, expected[column]):
                not_matching.append((member_name, column, col_type, expected[column]))
    return not_matching
