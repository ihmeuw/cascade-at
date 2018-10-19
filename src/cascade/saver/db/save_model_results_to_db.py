"""
Saves results from a model fit to a database which the EpiViz can access.
"""
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

from cascade.input_data.db.demographics import get_age_groups, get_years
from cascade.model.grids import unique_floats

try:
    from save_results._save_results import save_results_at
except ImportError:

    class DummySaveResults:
        def __getattr__(self, name):
            raise ImportError(f"Required package save_results not found")

    save_results_at = DummySaveResults()

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

DRAWS_INPUT_FILE_PATTERN = "all_draws.h5"

INTEGRAND_ID_TO_MEASURE_ID_DF = pd.DataFrame(
    [
        [0, 41],  # Susceptible incidence
        [1, 7],  # Remission
        [2, 9],  # Excess mortality rate
        [3, 16],  # Other cause mortality rate
        [4, 13],  # With-condition mortality rate
        [5, 39],  # Susceptible population fraction
        [6, 40],  # With Condition population fraction
        [7, 5],  # Prevalence
        [8, 42],  # Total Incidence
        [9, 15],  # Cause-specific mortality rate
        [10, 14],  # All-cause mortality rate
        [11, 12],  # Standardized mortality ratio
        [12, 11],  # Relative risk
    ],
    columns=["integrand_id", "measure_id"],
)


def _normalize_draws_df(draws_df, execution_context):
    """The database stores measure_id rather than integrand_id.

    Args:
        draws_df (DataFrame): The draws to normalize
        execution_context (ExecutionContext): The context to use when getting
                 demographic indexes for use in normalization.
    """

    draws = draws_df.merge(INTEGRAND_ID_TO_MEASURE_ID_DF, how="left", on="integrand_id").drop(columns=["integrand_id"])

    if not np.allclose(draws.time_lower, draws.time_upper):
        raise ValueError(
            "There are integrands over time intervals but we only " "know how to upload integrands for a point in time."
        )

    expected_years = sorted(get_years(execution_context))
    actual_years = sorted(unique_floats(draws.time_lower))
    if not np.allclose(expected_years, actual_years):
        raise ValueError("There are times in the avgint table that don't match GBD years")

    year_ids = pd.DataFrame({"year_id": expected_years, "year_float": np.array(expected_years, dtype=np.float64)})
    draws = pd.merge_asof(draws.sort_values("time_lower"), year_ids, left_on="time_lower", right_on="year_float").drop(
        ["time_lower", "time_upper", "year_float"], "columns"
    )

    age_groups = get_age_groups(execution_context)
    with_age_groups = pd.merge_asof(
        draws.sort_values("age_lower"), age_groups, left_on="age_lower", right_on="age_group_years_start"
    )

    merge_is_good = np.allclose(with_age_groups.age_lower, with_age_groups.age_group_years_start)
    merge_is_good = merge_is_good and np.allclose(with_age_groups.age_upper, with_age_groups.age_group_years_end)
    merge_is_good = merge_is_good and len(draws) == len(with_age_groups)
    if not merge_is_good:
        raise ValueError(
            "There are age_lowers or age_uppers in the avgint table that do not match GBD age group boundaries"
        )

    draws = with_age_groups.drop(["age_group_years_start", "age_group_years_end", "age_lower", "age_upper"], "columns")

    node_table = execution_context.dismodfile.node
    node_to_location = {r.node_id: r.c_location_id for _, r in node_table.iterrows()}

    draws["location_id"] = draws.node_id.apply(lambda nid: node_to_location[nid])

    draws["sex_id"] = draws.x_sex.apply(lambda x: {-0.5: 2, 0.5: 1}[x])
    return draws.drop(["node_id", "weight_id", "x_sex"], "columns")


def _write_temp_draws_file_and_upload_model_results(draws_df, execution_context):
    """

    Args:
        draws_df (pd.DataFrame): the draws data to upload
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the model_version_id returned by save_results

    """

    with tempfile.TemporaryDirectory() as tmpdirname:

        file_path = Path(tmpdirname) / DRAWS_INPUT_FILE_PATTERN

        draws_df.to_hdf(
            str(file_path),
            "draws",
            format="table",
            data_columns=["age_group_id", "location_id", "measure_id", "sex_id", "year_id"],
        )

        CODELOG.debug("Saving Results to DB")

        modelable_entity_id = execution_context.parameters.modelable_entity_id
        model_title = execution_context.parameters.model_title or ""
        measures_to_save = list(draws_df["measure_id"].unique())
        model_version_id = execution_context.parameters.model_version_id
        if "prod" in execution_context.parameters.database:
            db_env = "prod"
        else:
            db_env = "dev"

        model_version_id_df = save_results_at(
            tmpdirname,
            DRAWS_INPUT_FILE_PATTERN,
            modelable_entity_id,
            model_title,
            measures_to_save,
            model_version_id=model_version_id,
            db_env=db_env,
        )

        CODELOG.debug(f"model_version_id_df: {model_version_id_df.iloc[0, 0]}")

        return int(model_version_id_df.iloc[0, 0])


def save_model_results_to_db(draws_df, execution_context):
    """
    Gets the draws data normalized for the db and then uploads it.

    Args:
        draws_df (pd.DataFrame): the draws data to upload, after normalizing it
        execution_context (ExecutionContext): contains model id data

    Returns:
        (int) of the model_version_id returned by save_results

    """

    draws_df = _normalize_draws_df(draws_df, execution_context)

    model_version_id = _write_temp_draws_file_and_upload_model_results(draws_df, execution_context)

    return model_version_id
