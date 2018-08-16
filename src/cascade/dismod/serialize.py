"""
Converts the internal representation to a Dismod File.
"""
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum
from cascade.dismod.db.wrapper import _get_engine, DismodFile


LOGGER = logging.getLogger(__name__)


def model_to_dismod_file(model, filename):
    """
    This is a one-way translation from a model context to a new Dismod file.
    It assumes a lot. One location, no covariates, and more.

    Args:
        model_context (ModelContext): The one big object.

    """
    avgint_columns = dict()
    data_columns = dict()
    bundle_dismod_db = Path(filename)
    bundle_file_engine = _get_engine(bundle_dismod_db)
    bundle_fit = DismodFile(bundle_file_engine, avgint_columns, data_columns)

    # Standard Density table.
    density_enum = enum_to_dataframe(DensityEnum)
    densities = pd.DataFrame({"density_name": density_enum["name"]})
    bundle_fit.density = densities

    # Standard integrand naming scheme.
    all_integrands = default_integrand_names()
    bundle_fit.integrand = all_integrands

    bundle_fit.covariate = bundle_fit.empty_table("covariate")
    LOGGER.debug(f"Covariate types {bundle_fit.covariate.dtypes}")

    # Defaults, empty, b/c Brad makes them empty.
    bundle_fit.nslist = bundle_fit.empty_table("nslist")
    bundle_fit.nslist_pair = bundle_fit.empty_table("nslist_pair")
    bundle_fit.mulcov = bundle_fit.empty_table("mulcov")

    bundle_fit.log = _make_log_table()

    bundle_fit.node = _make_node_table(model)

    non_zero_rates = [rate.name for rate in model.rates if rate.parent_smooth or rate.child_smoothings]
    if "iota" in non_zero_rates:
        if "rho" in non_zero_rates:
            value = "iota_pos_rho_pos"
        else:
            value = "iota_pos_rho_zero"
    else:
        if "rho" in non_zero_rates:
            value = "iota_zero_rho_pos"
        else:
            value = "iota_zero_rho_zero"
    bundle_fit.option = pd.DataFrame(
        {
            "option_name": ["parent_node_name", "rate_case"],
            "option_value": [bundle_fit.node.iloc[0]["node_name"], value],
        }
    )

    bundle_fit.data = _make_data_table(model)

    # Include all data in the data_subset.
    bundle_fit.data_subset = pd.DataFrame({"data_id": np.arange(len(bundle_fit.data))})

    # Ages and times are used by Weight grids and smooth grids,
    # so pull all ages and times from those two objects in the
    # internal model. Skip weight grid here b/c assuming use constant.
    bundle_fit.age = _make_age_table(model)
    bundle_fit.time = _make_time_table(model)

    bundle_fit.weight, bundle_fit.weight_grid = simplest_weight()

    bundle_fit.prior, prior_id_func = make_prior_table(model, bundle_fit.density)
    bundle_fit.smooth, bundle_fit.smooth_grid, smooth_id_func = make_smooth_and_smooth_grid_tables(
        model, bundle_fit.age, bundle_fit.time, prior_id_func
    )

    def integrand_id_func(name):
        return int(bundle_fit.integrand.query("integrand_name==@name").integrand_id)

    # The avgint needs to be translated.
    bundle_fit.avgint = _make_avgint_table(model, integrand_id_func)

    bundle_fit.rate = _make_rate_table(model, smooth_id_func)

    return bundle_fit


def enum_to_dataframe(enum_name):
    """Given an enum, return a dataframe with two columns, name and value."""
    return pd.DataFrame.from_records(
        np.array(
            [(measure, enum_value.value) for (measure, enum_value) in enum_name.__members__.items()],
            dtype=np.dtype([("name", object), ("value", np.int)]),
        )
    )


def default_integrand_names():
    # Converting an Enum to a DataFrame with specific parameters
    integrands = enum_to_dataframe(IntegrandEnum)
    df = pd.DataFrame({"integrand_name": integrands["name"]})
    df["minimum_meas_cv"] = 0.0
    return df


def _make_log_table():
    return pd.DataFrame(
        {
            "message_type": ["command"],
            "table_name": np.array([None], dtype=np.object),
            "row_id": np.NaN,
            "unix_time": int(round(time.time())),
            "message": ["fit_no_covariates.py"],
        }
    )


def _make_node_table(context):
    # Assume we have one location, so no parents.
    # If we had a hierarchy, that would be used to determine parents.
    unique_locations = context.input_data.observations["location_id"].unique()
    assert len(unique_locations) == 1
    assert unique_locations[0] == context.parameters.location_id

    return pd.DataFrame({"node_name": unique_locations.astype(int).astype(str), "parent": np.array([np.NaN])})


def _make_data_table(context):
    observations = observations_to_data(context.input_data.observations)
    constraints = observations_to_data(context.input_data.constraints, hold_out=1)
    total_data = pd.concat([observations, constraints], ignore_index=True)
    # Why a unique string name?
    total_data["data_name"] = total_data.index.astype(str)
    return total_data


def simplest_weight():
    """Defines one weight for everything by defining it on one age-time point."""
    weight = pd.DataFrame({"weight_name": ["constant"], "n_age": [1], "n_time": [1]})
    weight_grid = pd.DataFrame({"weight_id": [0], "age_id": [0], "time_id": [0], "weight": [1.0]})
    return weight, weight_grid


def observations_to_data(observations_df, hold_out=0):
    """Turn an internal format into a Dismod format."""
    # Don't make the data_name here because could convert multiple observations.
    return pd.DataFrame(
        {
            "integrand_id": observations_df["measure"].apply(lambda x: IntegrandEnum[x].value),
            # Assumes one location_id.
            "node_id": 0,
            # Density is an Enum at this point.
            "density_id": observations_df["density"].apply(lambda x: x.value),
            # Translate weight from string
            "weight_id": 0,
            "age_lower": observations_df["age_start"],
            "age_upper": observations_df["age_end"],
            "time_lower": observations_df["year_start"].astype(np.float),
            "time_upper": observations_df["year_end"],
            "meas_value": observations_df["mean"],
            "meas_std": observations_df["standard_error"],
            "eta": np.NaN,
            "nu": np.NaN,
            "hold_out": hold_out,
        }
    )


def collect_priors(context):
    priors = set()

    for rate in context.rates:
        for smooth in rate.smoothings:
            priors.update([p for g in smooth.prior_grids for p in g.priors])

    return priors


def collect_ages_or_times(context, to_collect="ages"):
    if to_collect not in ("ages", "times"):
        raise ValueError("to_collect must be either 'ages' or 'times'")

    values = []

    for rate in context.rates:
        for smooth in rate.smoothings:
            if to_collect == "ages":
                value = smooth.grid.ages
            else:
                value = smooth.grid.times
            values.extend(value)

    # Extreme values from the input data must also appear in the age/time table
    if to_collect == "ages":
        values.append(
            np.max([np.max(context.input_data.observations.age_end), np.max(context.input_data.constraints.age_end)])
        )
        values.append(
            np.max(
                [np.max(context.input_data.observations.age_start), np.max(context.input_data.constraints.age_start)]
            )
        )
    else:
        values.append(
            np.max([np.max(context.input_data.observations.year_end), np.max(context.input_data.constraints.year_end)])
        )
        values.append(
            np.max(
                [np.max(context.input_data.observations.year_start), np.max(context.input_data.constraints.year_start)]
            )
        )

    values = np.array(values)
    uniqued_values = np.unique(values.round(decimals=14), return_index=True)[1]

    return values[uniqued_values]


def _make_age_table(context):
    ages = collect_ages_or_times(context, "ages")
    age_df = pd.DataFrame(ages, columns=["age"])
    age_df["age_id"] = age_df.index

    return age_df


def _make_time_table(context):
    times = collect_ages_or_times(context, "times")
    time_df = pd.DataFrame(times, columns=["time"])
    time_df["time_id"] = time_df.index

    return time_df


def _make_avgint_table(context, integrand_id_func):
    rows = []
    for integrand in context.outputs.integrands:
        if integrand.active:
            rows.append(
                {
                    "integrand_id": integrand_id_func(integrand.name),
                    "age_lower": integrand.age_lower,
                    "age_upper": integrand.age_upper,
                    "time_lower": integrand.time_lower,
                    "time_upper": integrand.time_upper,
                    # Assuming using the first set of weights, which is constant.
                    "weight_id": 0,
                    # Assumes one location_id.
                    "node_id": 0,
                }
            )
    return pd.DataFrame(rows)


def _prior_to_row(prior):
    row = {
        "prior_name": None,
        "density": None,
        "lower": np.nan,
        "upper": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "eta": np.nan,
        "nu": np.nan,
    }
    row.update(prior.parameters())
    row["density_name"] = row["density"]
    del row["density"]
    return row


def make_prior_table(context, density_table):
    priors = list(collect_priors(context))

    prior_table = pd.DataFrame([_prior_to_row(p) for p in priors])
    prior_table["prior_id"] = prior_table.index
    prior_table.loc[prior_table.prior_name.isnull(), "prior_name"] = prior_table.loc[
        prior_table.prior_name.isnull(), "prior_id"
    ].apply(lambda pid: f"prior_{pid}")

    prior_table = pd.merge(prior_table, density_table, on="density_name")
    prior_table["prior_id"] = prior_table.index

    def prior_id_func(prior):
        return priors.index(prior)

    return prior_table.drop("density_name", "columns"), prior_id_func


def make_smooth_grid_table(smooth, prior_id_func):
    grid = smooth.grid

    rows = []
    if grid is not None:
        for age in grid.ages:
            for year in grid.times:
                row = {"age": age, "time": year, "const_value": np.nan}
                if smooth.value_priors:
                    row["value_prior_id"] = prior_id_func(smooth.value_priors[age, year].prior)
                else:
                    row["value_prior_id"] = None
                if smooth.d_age_priors:
                    row["dage_prior_id"] = prior_id_func(smooth.d_age_priors[age, year].prior)
                else:
                    row["dage_prior_id"] = None
                if smooth.d_time_priors:
                    row["dtime_prior_id"] = prior_id_func(smooth.d_time_priors[age, year].prior)
                else:
                    row["dtime_prior_id"] = None
                rows.append(row)

    return pd.DataFrame(
        rows, columns=["age", "time", "const_value", "value_prior_id", "dage_prior_id", "dtime_prior_id"]
    )


def _smooth_row(name, smooth, grid, prior_id_func):
    if smooth.value_priors and smooth.value_priors.hyper_prior:
        mulstd_value_prior_id = prior_id_func(smooth.value_priors.hyper_prior)
    else:
        mulstd_value_prior_id = np.nan
    if smooth.d_age_priors and smooth.d_age_priors.hyper_prior:
        mulstd_dage_prior_id = prior_id_func(smooth.d_age_priors.hyper_prior)
    else:
        mulstd_dage_prior_id = np.nan
    if smooth.d_time_priors and smooth.d_time_priors.hyper_prior:
        mulstd_dtime_prior_id = prior_id_func(smooth.d_time_priors.hyper_prior)
    else:
        mulstd_dtime_prior_id = np.nan

    return {
        "smooth_name": name,
        "n_age": len(grid.age_id.unique()),
        "n_time": len(grid.time_id.unique()),
        "mulstd_value_prior_id": mulstd_value_prior_id,
        "mulstd_dage_prior_id": mulstd_dage_prior_id,
        "mulstd_dtime_prior_id": mulstd_dtime_prior_id,
    }


def make_smooth_and_smooth_grid_tables(context, age_table, time_table, prior_id_func):
    grid_tables = []
    smooths = []
    for rate in context.rates:
        for smooth in rate.child_smoothings + [rate.parent_smooth] if rate.parent_smooth else []:
            grid_table = make_smooth_grid_table(smooth, prior_id_func)
            smooths.append(smooth)
            grid_table["smooth_id"] = len(smooths)
            grid_tables.append(grid_table)
    smooths = list(smooths)

    if grid_tables:
        grid_table = pd.concat(grid_tables)
        grid_table = pd.merge_asof(grid_table.sort_values("age"), age_table, on="age").drop("age", "columns")
        grid_table = pd.merge_asof(grid_table.sort_values("time"), time_table, on="time").drop("time", "columns")
    else:
        grid_table = pd.DataFrame()

    smooth_table = pd.DataFrame(
        [_smooth_row(f"smooth_{i}", smooth, grid_table, prior_id_func) for i, smooth in enumerate(smooths)]
    )

    def smooth_id_func(smooth):
        return smooths.index(smooth)

    return smooth_table, grid_table, smooth_id_func


def _make_rate_table(context, smooth_id_func):
    rows = []
    for rate in context.rates:
        if rate.parent_smooth or rate.child_smoothings:
            if len(rate.child_smoothings) > 1:
                raise NotImplementedError("Multiple child smoothings not supported yet")

            rows.append(
                {
                    "rate_name": rate.name,
                    "parent_smooth_id": smooth_id_func(rate.parent_smooth) if rate.parent_smooth else np.NaN,
                    "child_smooth_id": smooth_id_func(rate.child_smoothings[0]) if rate.child_smoothings else np.NaN,
                    "child_nslist_id": np.NaN,
                }
            )
    return pd.DataFrame(rows)


def read_predict(db_path):
    avgint_columns = dict()
    data_columns = dict()
    bundle_dismod_db = Path(db_path)
    bundle_file_engine = _get_engine(bundle_dismod_db)
    bundle_fit = DismodFile(bundle_file_engine, avgint_columns, data_columns)

    desired_outputs = bundle_fit.avgint

    # Use the integrand table to convert the integrand_id into an integrand_name.
    integrand_names = bundle_fit.integrand
    desired_outputs = desired_outputs.merge(integrand_names, left_on="integrand_id", right_on=integrand_names.index)

    # Associate the result with the desired integrand.
    prediction = bundle_fit.predict.merge(desired_outputs, left_on="avgint_id", right_on=desired_outputs.index)
    return prediction


def read_prevalence(prediction):
    return prediction[prediction["integrand_name"] == "prevalence"][["avg_integrand", "time_lower", "age_lower"]]
