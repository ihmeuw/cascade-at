import numpy as np
import pandas as pd
from numpy import dtype

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


EXPECTED_TYPES = dict(
    age_specific_death_rate=[
        ('location', dtype('int64')),
        ('sex_id', dtype('int64')),
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('float64')),
        ('time_upper', dtype('float64')),
        ('integrand', dtype('O')),
        ('mean', dtype('float64')),
        ('std', dtype('float64')),
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
        ('integrand', dtype('O')),
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('int64')),
        ('time_upper', dtype('int64')),
        ('location', dtype('int64')),
        ('sex_id', dtype('int64')),
    ],
    bundle=[
        ('seq', dtype('int64')),
        ('measure', dtype('O')),
        ('sex_id', dtype('int64')),
        ('mean', dtype('float64')),
        ('lower', dtype('float64')),
        ('upper', dtype('float64')),
        ('hold_out', dtype('int64')),
        ('age_lower', dtype('float64')),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('int64')),
        ('time_upper', dtype('int64')),
        ('location_id', dtype('int64')),
    ],
    cause_specific_mortality_rate=[
        ("location_id", dtype("int64")),
        ("sex_id", dtype("int64")),
        ("mean", dtype("float64")),
        ("lower", dtype("float64")),
        ("upper", dtype("float64")),
        ("age_lower", dtype("float64")),
        ('age_upper', dtype('float64')),
        ('time_lower', dtype('float64')),
        ('time_upper', dtype('float64')),
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
        ('location_id', dtype('int64')),
    ]
)
"""These are the data frames and column data types that define
the raw input data. If you want to mock input data or get it
from another source, it has to have these columns."""


def validate_input_data_types(input_data):
    """This is a gating function that insists the input data have only certain
    data sets with certain columns and data types."""
    not_matching = list()
    seen = set()
    # Search for Pandas dataframes in the input_data object.
    for member_name in dir(input_data):
        dtypes = dtype_of_member(input_data, member_name)
        if dtypes is not None:
            seen.add(member_name)
            not_matching.extend(check_one_member(dtypes, member_name))
        # else it isn't a Pandas DataFrame.
    dataframes_that_werent_checked = seen - set(EXPECTED_TYPES.keys())
    assert not dataframes_that_werent_checked
    return not_matching


def dtype_of_member(input_data, member_name):
    """Find the dtypes of columns of a pd.DataFrame, if it's a DataFrame."""
    member = getattr(input_data, member_name)
    # Ignore dataframes of length 0. They have no data types for columns usually.
    if isinstance(member, pd.DataFrame) and len(member) > 0:
        dtypes = member.dtypes
    elif isinstance(member, dict) and member:
        # Look at the first dictionary in a dictionary of dataframes. Assume
        # the rest are the same. The way we make them, this should be the case.
        dict_member = next(iter(member.values()))
        if isinstance(dict_member, pd.DataFrame) and len(dict_member) > 0:
            dtypes = dict_member.dtypes
        else:
            dtypes = None
    else:
        dtypes = None
    return dtypes


def check_one_member(dtypes_of_input, member_name):
    """
    Check dtypes of input data against expected dtypes.

    Args:
        dtypes_of_input (List[(column name, dtype)]): List of data types
        member_name (str): Name of a dataframe.

    Returns:
        List of errors. An empty list means nothing is wrong.
    """
    not_matching = list()
    variable_dype = dict(zip(dtypes_of_input.index.tolist(), dtypes_of_input.tolist()))
    if member_name not in EXPECTED_TYPES:
        CODELOG.info(f"{member_name} {list(zip(dtypes_of_input.index.tolist(), dtypes_of_input.tolist()))}")
        not_matching.append(f"Cannot check types for dataframe named {member_name} during validation.")
    else:
        expected = dict(EXPECTED_TYPES[member_name])
        for column, col_type in variable_dype.items():
            col_name = f"The DataFrame column {member_name}.{column}"
            if column not in expected:
                not_matching.append(f"{col_name} was unexpectedly added to input data.")
            else:
                want_float = expected[column] == dtype("float64")
                if want_float:
                    isnt_int_or_float = col_type not in {dtype("float64"), dtype("int64")}
                    if isnt_int_or_float:
                        not_matching.append(f"{col_name} should be a float but is {col_type}.")
                else:
                    dtype_subset = np.issubdtype(col_type, expected[column])
                    # issubdtype failed for comparing int64 to int64 and float64 to float64.
                    same_name = col_type.name == expected[column].name
                    if not (dtype_subset or same_name):
                        CODELOG.debug(f"check failed {col_type.descr} {expected[column].descr}")
                        msg = f"{col_name} should be {expected[column]} but is {col_type}"
                        not_matching.append(msg)
            # else it's fine so append nothing.
    return not_matching
