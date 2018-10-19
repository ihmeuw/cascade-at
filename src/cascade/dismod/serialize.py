"""
Converts the internal representation to a Dismod File.
"""
from numbers import Real
import time
import sys

import numpy as np
import pandas as pd

from cascade.dismod.db.metadata import IntegrandEnum, DensityEnum
from cascade.dismod.db.wrapper import DismodFile
from cascade.model.priors import Constant
from cascade.model.grids import unique_floats
from cascade.input_data.db.locations import get_location_hierarchy_from_gbd

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def model_to_dismod_file(model, execution_context):
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
    bundle_fit.integrand["minimum_meas_cv"] = model.parameters.minimum_meas_cv

    bundle_fit.covariate = bundle_fit.empty_table("covariate")
    CODELOG.debug(f"Covariate types {bundle_fit.covariate.dtypes}")

    # Defaults, empty, b/c Brad makes them empty.
    bundle_fit.nslist = bundle_fit.empty_table("nslist")
    bundle_fit.nslist_pair = bundle_fit.empty_table("nslist_pair")
    bundle_fit.mulcov = bundle_fit.empty_table("mulcov")

    bundle_fit.log = make_log_table()

    bundle_fit.node, location_to_node_func = make_node_table(execution_context)

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

    bundle_fit.data = make_data_table(model, bundle_fit.node)

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

    # Given average integrand cases by name, convert them to average integrand
    # cases by ID and save. Any covariates have to exist at this point.
    bundle_fit.avgint = make_avgint_table(model, integrand_id_func, location_to_node_func)

    bundle_fit.rate, rate_id_func = make_rate_table(model, smooth_id_func)

    bundle_fit.covariate, bundle_fit.mulcov, covariate_id_func = make_covariate_table(
        model, smooth_id_func, rate_id_func, integrand_to_id
    )

    bundle_fit.option = make_option_table(model)

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
    command_name = " ".join(sys.argv)
    return pd.DataFrame(
        {
            "message_type": ["command"],
            "table_name": np.array([None], dtype=np.object),
            "row_id": np.NaN,
            "unix_time": int(round(time.time())),
            "message": [command_name],
        }
    )


def rec_build_nodes_table(node, parent):
    parent_id = parent.id if parent is not None else np.NaN
    result = [{"node_name": node.info["location_name_short"], "c_location_id": node.id, "parent": parent_id}]
    for child in node.level_n_descendants(1):
        result += rec_build_nodes_table(child, node)
    return result


def make_node_table(execution_context):
    locations = get_location_hierarchy_from_gbd(execution_context)
    table = pd.DataFrame(rec_build_nodes_table(locations.root, None), columns=["node_name", "parent", "c_location_id"])
    table["node_id"] = table.index

    def location_to_node_func(location_id):
        if np.isnan(location_id):
            return np.nan
        return np.where(table.c_location_id == location_id)[0][0]

    table["parent"] = table.parent.apply(location_to_node_func)

    return table, location_to_node_func


def make_data_table(context, node_table):
    total_data = []
    if context.input_data.observations is not None:
        # It's OK for observations to be None if we are running a prediction.
        total_data.append(observations_to_data(context.input_data.observations, node_table))
    if context.input_data.constraints is not None:
        # While constraints are defined as smoothings on rates, these same
        # data values are put into measurement data as hold-outs so that they
        # can be visualized with the data and residuals.
        total_data.append(observations_to_data(context.input_data.constraints, node_table, hold_out=1))

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


def observations_to_data(observations_df, node_table, hold_out=0):
    """Turn an internal format into a Dismod format."""
    # Don't make the data_name here because could convert multiple observations.
    observations_df = observations_df.reset_index()
    observations_df["node_id"] = observations_df.merge(node_table,
                                                       left_on="node_id",
                                                       right_on=node_table.c_location_id
                                                       ).node_id
    return pd.DataFrame(
        {
            "integrand_id": observations_df["measure"].apply(lambda x: IntegrandEnum[x].value),
            "node_id": observations_df.node_id,
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

    for smooth in smooth_iter(context):
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

    if to_collect == "ages":
        value = np.concatenate([context.average_integrand_cases.age_lower, context.average_integrand_cases.age_upper])
    else:
        value = np.concatenate([context.average_integrand_cases.time_lower, context.average_integrand_cases.time_upper])
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


def integrand_to_id(integrand):
    """
    This is the general function from integrand to its index in the db.

    Args:
        integrand (str): The name of the integrand.
    """
    return IntegrandEnum[integrand].value


def make_avgint_table(context, integrand_id_func, location_to_node_func):
    if context.average_integrand_cases is not None:
        df = context.average_integrand_cases.copy()
        df["integrand_id"] = df.integrand_name.apply(integrand_id_func)
        df["node_id"] = df.node_id.apply(location_to_node_func)
        return df.drop(columns=["integrand_name"])
    else:
        covariate_names = [cov_obj.name for cov_obj in context.input_data.covariates]
        all_avgint_columns = ["integrand_id", "age_lower", "age_upper",
                              "time_lower", "time_upper", "weight_id",
                              "node_id"] + covariate_names
        return pd.DataFrame([], columns=all_avgint_columns)


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


def covariate_multiplier_iter(context):
    """
    Covariate multipliers are stored in three places of the context.
    This iterates through those three places. The same covariate multiplier
    instance can be attached to more than one of those three places, and each
    time it creates a different covariate multiplier set of model variables.
    """
    # α according to Dismod-AT
    for rate in context.rates:
        for rate_mul in rate.covariate_multipliers:
            yield rate_mul, "rate_value", rate

    for integrand in context.integrand_covariate_multipliers.values():
        # β
        for val_mul in integrand.value_covariate_multipliers:
            yield val_mul, "meas_value", integrand
        # γ
        for std_mul in integrand.std_covariate_multipliers:
            yield std_mul, "meas_std", integrand


def smooth_iter(context):
    """Iterate over every smooth in the context."""
    for rate in context.rates:
        for smooth in [s for _, s in rate.child_smoothings] + [rate.parent_smooth] if rate.parent_smooth else []:
            yield smooth

    for cov_multiplier, _, _ in covariate_multiplier_iter(context):
        yield cov_multiplier.smooth


def make_smooth_and_smooth_grid_tables(context, age_table, time_table, prior_id_func):
    grid_tables = []
    smooths = []
    smooth_rows = []

    for smooth in smooth_iter(context):
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
    rate_to_id = {}
    rows = []
    for rate_id, rate in enumerate(context.rates):
        if len(rate.child_smoothings) > 1:
            raise NotImplementedError("Multiple child smoothings not supported yet")

        rows.append(
            {
                "rate_id": rate_id,
                "rate_name": rate.name,
                "parent_smooth_id": smooth_id_func(rate.parent_smooth) if rate.parent_smooth else np.NaN,
                "child_smooth_id": smooth_id_func(rate.child_smoothings[0][1]) if rate.child_smoothings else np.NaN,
                "child_nslist_id": np.NaN,
            }
        )
        rate_to_id[rate] = rate_id

    def rate_id_func(rate):
        return rate_to_id[rate]

    return pd.DataFrame(rows), rate_id_func


def make_covariate_table(context, smooth_id_func, rate_id_func, integrand_id_func):
    cols = context.input_data.covariates
    covariate_columns = pd.DataFrame(
        {
            "covariate_id": np.arange(len(cols)),
            "covariate_name": [col.name for col in cols],
            "reference": np.array([col.reference for col in cols], dtype=np.float),
            "max_difference": np.array([col.max_difference for col in cols], dtype=np.float),
        }
    )
    if covariate_columns["reference"].isnull().any():
        null_references = list()
        for check_ref_col in cols:
            if not isinstance(check_ref_col.reference, Real):
                null_references.append(check_ref_col.name)
        raise RuntimeError(f"Covariate columns without reference values {null_references}")

    def cov_col_id_func(query_column):
        return cols.index(query_column)

    cm_rows = []
    # The kinds are described here:
    # https://bradbell.github.io/dismod_at/doc/avg_integrand.htm
    for cidx, mul_type in enumerate(covariate_multiplier_iter(context)):
        cov_mul, kind, rate_or_integrand = mul_type
        if kind == "rate_value":
            rate_id = rate_id_func(rate_or_integrand)
            integrand_id = np.NaN
        else:
            rate_id = np.NaN
            integrand_id = integrand_id_func(rate_or_integrand.name)

        cm_rows.append(
            dict(
                mulcov_id=cidx,
                mulcov_type=kind,
                rate_id=rate_id,
                integrand_id=integrand_id,
                covariate_id=cov_col_id_func(cov_mul.column),
                smooth_id=smooth_id_func(cov_mul.smooth),
            )
        )
    mul_cov = pd.DataFrame(
        cm_rows, columns=["mulcov_id", "mulcov_type", "rate_id", "integrand_id", "covariate_id", "smooth_id"]
    )

    return covariate_columns, mul_cov, cov_col_id_func


def make_option_table(context):
    options = {
        "rate_case": context.parameters.rate_case,
        "parent_node_id": "0",
        "print_level_fixed": "5",
        "ode_step_size": "1",
        "quasi_fixed": "false",
    }

    return pd.DataFrame([{"option_name": k, "option_value": v} for k, v in sorted(options.items())])
