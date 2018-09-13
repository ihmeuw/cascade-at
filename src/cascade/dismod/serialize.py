"""
Converts the internal representation to a Dismod File.
"""
import logging
import warnings
import time

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum
from cascade.dismod.db.wrapper import DismodFile
from cascade.model.priors import Constant
from cascade.model.grids import unique_floats


LOGGER = logging.getLogger(__name__)


def model_to_dismod_file(model):
    """
    This is a one-way translation from a model context to a new Dismod file.
    It assumes a lot. One location, no covariates, and more.

    Args:
        model_context (ModelContext): The one big object.

    """
    bundle_fit = DismodFile()

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

    bundle_fit.log = make_log_table()

    bundle_fit.node = make_node_table(model)

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

    bundle_fit.data = make_data_table(model)

    # Include all data in the data_subset.
    bundle_fit.data_subset = pd.DataFrame({"data_id": np.arange(len(bundle_fit.data))})

    # Ages and times are used by Weight grids and smooth grids,
    # so pull all ages and times from those two objects in the
    # internal model. Skip weight grid here b/c assuming use constant.
    bundle_fit.age = make_age_table(model)
    bundle_fit.time = make_time_table(model)

    bundle_fit.weight, bundle_fit.weight_grid = simplest_weight()

    bundle_fit.prior, prior_id_func = make_prior_table(model, bundle_fit.density)
    bundle_fit.smooth, bundle_fit.smooth_grid, smooth_id_func = make_smooth_and_smooth_grid_tables(
        model, bundle_fit.age, bundle_fit.time, prior_id_func
    )

    def integrand_id_func(name):
        return int(bundle_fit.integrand.query("integrand_name==@name").integrand_id)

    # The avgint needs to be translated.
    bundle_fit.avgint = make_avgint_table(model, integrand_id_func)

    bundle_fit.rate = make_rate_table(model, smooth_id_func)

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


def make_log_table():
    return pd.DataFrame(
        {
            "message_type": ["command"],
            "table_name": np.array([None], dtype=np.object),
            "row_id": np.NaN,
            "unix_time": int(round(time.time())),
            "message": ["fit_no_covariates.py"],
        }
    )


def make_node_table(context):
    # Assume we have one location, so no parents.
    # If we had a hierarchy, that would be used to determine parents.
    if context.input_data.observations is not None and not context.input_data.observations.empty:
        unique_locations = context.input_data.observations["location_id"].unique()
        assert len(unique_locations) == 1
    else:
        warnings.warn("No observations in model, falling back to location_id in parameters")
        unique_locations = np.array([context.parameters.location_id])

    return pd.DataFrame({"node_name": unique_locations.astype(int).astype(str), "parent": np.array([np.NaN])})


def make_data_table(context):
    total_data = []
    if context.input_data.observations is not None:
        total_data.append(observations_to_data(context.input_data.observations))
    if context.input_data.constraints is not None:
        total_data.append(observations_to_data(context.input_data.constraints, hold_out=1))

    if total_data:
        total_data = pd.concat(total_data, ignore_index=True)
        # Why a unique string name?
        total_data["data_name"] = total_data.index.astype(str)
    else:
        total_data = pd.DataFrame(
            columns=[
                "integrand_id",
                "node_id",
                "density_id",
                "weight_id",
                "age_lower",
                "age_upper",
                "time_lower",
                "time_upper",
                "meas_value",
                "meas_std",
                "eta",
                "nu",
                "hold_out",
                "data_name",
            ]
        )
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
            for grid_name in ["d_age_priors", "d_time_priors", "value_priors"]:
                grid = getattr(smooth, grid_name)
                if grid:
                    ps = grid.priors
                    if grid_name == "value_priors":
                        # Constants on the value don't actually go in the
                        # prior table, so exclude them
                        ps = [p for p in ps if not isinstance(p, Constant)]
                    priors.update(ps)

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
    if to_collect == "ages" and context.input_data.ages:
        values.append(np.max(list(context.input_data.ages)))
        values.append(np.min(list(context.input_data.ages)))
    elif context.input_data.times:
        values.append(np.max(list(context.input_data.times)))
        values.append(np.min(list(context.input_data.times)))

    return sorted(unique_floats(values))


def make_age_table(context):
    ages = collect_ages_or_times(context, "ages")
    age_df = pd.DataFrame(ages, columns=["age"], dtype=float)
    age_df["age_id"] = age_df.index

    return age_df


def make_time_table(context):
    times = collect_ages_or_times(context, "times")
    time_df = pd.DataFrame(times, columns=["time"], dtype=float)
    time_df["time_id"] = time_df.index

    return time_df


def make_avgint_table(context, integrand_id_func):
    rows = []
    for integrand in context.outputs.integrands:
        if integrand.grid is not None:
            for a in integrand.grid.ages:
                for t in integrand.grid.times:
                    rows.append(
                        {
                            "integrand_id": integrand_id_func(integrand.name),
                            "age_lower": a,
                            "age_upper": a,
                            "time_lower": t,
                            "time_upper": t,
                            # Assuming using the first set of weights, which is constant.
                            "weight_id": 0,
                            # Assumes one location_id.
                            "node_id": 0,
                        }
                    )
    return pd.DataFrame(
        rows, columns=["integrand_id", "age_lower", "age_upper", "time_lower", "time_upper", "weight_id", "node_id"]
    )


def _prior_row(prior):
    row = {
        "prior_name": prior.name,
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

    if row["eta"] is None:
        # For some distributions eta is a required parameter but for others
        # it is nullable and represents an offset to be used during
        # optimization. This let's us have None represent the missing
        # value in python
        row["eta"] = np.nan

    del row["density"]
    return row


def make_prior_table(context, density_table):
    priors = sorted(collect_priors(context))

    prior_table = pd.DataFrame(
        [_prior_row(p) for p in priors],
        columns=["prior_name", "density_name", "lower", "upper", "mean", "std", "eta", "nu"],
    )
    prior_table["prior_id"] = prior_table.index
    prior_table.loc[prior_table.prior_name.isnull(), "prior_name"] = prior_table.loc[
        prior_table.prior_name.isnull(), "prior_id"
    ].apply(lambda pid: f"prior_{pid}")

    prior_table["prior_id"] = prior_table.index
    prior_table = pd.merge(prior_table, density_table, on="density_name")
    # Make sure the index still matches the order in the priors list
    prior_table = prior_table.sort_values(by="prior_id").reset_index(drop=True)

    def prior_id_func(prior):
        return priors.index(prior)

    return prior_table.drop("density_name", "columns"), prior_id_func


def make_smooth_grid_table(smooth, prior_id_func):
    grid = smooth.grid

    rows = []
    if grid is not None:
        for year in grid.times:
            for age in grid.ages:
                row = {"age": float(age), "time": float(year), "const_value": np.nan}
                if smooth.value_priors:
                    prior = smooth.value_priors[age, year].prior
                    if isinstance(prior, Constant):
                        row["const_value"] = prior.value
                        row["value_prior_id"] = np.nan
                    else:
                        row["value_prior_id"] = prior_id_func(prior)
                else:
                    row["value_prior_id"] = np.nan
                if smooth.d_age_priors:
                    row["dage_prior_id"] = prior_id_func(smooth.d_age_priors[age, year].prior)
                else:
                    row["dage_prior_id"] = np.nan
                if smooth.d_time_priors:
                    row["dtime_prior_id"] = prior_id_func(smooth.d_time_priors[age, year].prior)
                else:
                    row["dtime_prior_id"] = np.nan
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
    smooth_rows = []
    for rate in context.rates:
        for smooth in rate.child_smoothings + [rate.parent_smooth] if rate.parent_smooth else []:
            grid_table = make_smooth_grid_table(smooth, prior_id_func)
            grid_table["smooth_id"] = len(smooths)
            grid_table = pd.merge_asof(grid_table.sort_values("age"), age_table, on="age").drop("age", "columns")
            grid_table = pd.merge_asof(grid_table.sort_values("time"), time_table, on="time").drop("time", "columns")
            grid_table = grid_table.sort_values(["time_id", "age_id"])

            if smooth.name is None:
                name = f"smooth_{len(smooths)}"
            else:
                name = smooth.name
            smooth_rows.append(_smooth_row(name, smooth, grid_table, prior_id_func))
            smooths.append(smooth)
            grid_tables.append(grid_table)

    if grid_tables:
        grid_table = pd.concat(grid_tables).reset_index(drop=True)
        grid_table["smooth_grid_id"] = grid_table.index
    else:
        grid_table = pd.DataFrame(
            columns=[
                "age_id",
                "time_id",
                "const_value",
                "value_prior_id",
                "dage_prior_id",
                "dtime_prior_id",
                "smooth_grid_id",
                "smooth_id",
            ]
        )

    smooth_table = pd.DataFrame(
        smooth_rows,
        columns=[
            "smooth_name",
            "n_age",
            "n_time",
            "mulstd_value_prior_id",
            "mulstd_dage_prior_id",
            "mulstd_dtime_prior_id",
        ],
    )

    def smooth_id_func(smooth):
        return smooths.index(smooth)

    return smooth_table, grid_table, smooth_id_func


def make_rate_table(context, smooth_id_func):
    rows = []
    for rate in context.rates:
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
