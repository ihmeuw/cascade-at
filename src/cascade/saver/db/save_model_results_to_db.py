"""
Saves results from a model fit to a database which the EpiViz can access.
"""
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

from cascade.input_data.db.demographics import get_age_groups, get_years
from cascade.model.grids import unique_floats
from cascade.core.db import save_results


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
    covariate_table = execution_context.dismodfile.covariate
    try:
        sex_index = int(covariate_table[covariate_table.covariate_name == "sex"].covariate_id.iloc[0])
    except KeyError as ke:
        raise RuntimeError(f"Output from Dismod-AT lacks a sex column, so upload not possible.") from ke

    sex_column = f"x_{sex_index}"
    ids = pd.DataFrame({"sex_id": [1, 2, 3], "x_sex": [0.5, -0.5, 0.0]}).sort_values(by=["x_sex"])
    draws = pd.merge_asof(draws.sort_values(by=[sex_column]), ids, left_on=sex_column, right_on="x_sex")
    # Remove covariates from draws to upload.
    to_drop = ["node_id", "weight_id"] + [str(cov_col) for cov_col in draws.columns if cov_col.startswith("x_")]
    return draws.drop(to_drop, "columns")


def _write_temp_draws_file_and_upload_model_results(draws_df, execution_context, saver=None):
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

        modelable_entity_id = execution_context.parameters.modelable_entity_id
        model_title = execution_context.parameters.model_title or ""
        measures_to_save = list(draws_df["measure_id"].unique())
        model_version_id = execution_context.parameters.model_version_id
        gbd_round_id = execution_context.parameters.gbd_round_id
        year_ids = list(draws_df.year_id.unique())
        if "prod" in execution_context.parameters.database:
            db_env = "prod"
        else:
            db_env = "dev"

        CODELOG.debug(f"Saving Results to DB years {year_ids} "
                      f"measure_id {draws_df.measure_id.unique()} "
                      f"age_group_id {draws_df.age_group_id.unique()} "
                      f"round {gbd_round_id} env {db_env} mvid {model_version_id} ")

        saver = saver if saver else save_results.save_results_at
        model_version_id_df = saver(
            tmpdirname,
            DRAWS_INPUT_FILE_PATTERN,
            modelable_entity_id,
            model_title,
            measures_to_save,
            year_id=year_ids,
            model_version_id=model_version_id,
            db_env=db_env,
            gbd_round_id=gbd_round_id,
            sex_id=list(draws_df.sex_id.unique()),
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
