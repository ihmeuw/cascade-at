import pandas as pd
import numpy as np

from cascade.dismod.db.wrapper import DismodFile


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

    values = np.array(values)
    uniqued_values = np.unique(values.round(decimals=14), return_index=True)[1]

    return values[uniqued_values]


def dismodfile_from_model_context(context):
    dm = DismodFile(None, {}, {})
    dm.make_densities()

    dm.age = make_age_table(context)
    dm.time = make_time_table(context)

    dm.prior, prior_objects = make_prior_table(context, dm.density)
    dm.smooth, dm.smooth_grid = make_smooth_and_smooth_grid_tables(context, dm.age, dm.time, prior_objects)

    return dm


def make_age_table(context):
    ages = collect_ages_or_times(context, "ages")
    age_df = pd.DataFrame(ages, columns=["age"])
    age_df["age_id"] = age_df.index

    return age_df


def make_time_table(context):
    times = collect_ages_or_times(context, "times")
    time_df = pd.DataFrame(times, columns=["time"])
    time_df["time_id"] = time_df.index

    return time_df


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

    return prior_table.drop("density_name", "columns"), priors


def make_smooth_grid_table(smooth, prior_objects):
    grid = smooth.grid

    rows = []
    if grid is not None:
        for age in grid.ages:
            for time in grid.times:
                row = {"age": age, "time": time, "const_value": np.nan}
                if smooth.value_priors:
                    row["value_prior_id"] = prior_objects.index(smooth.value_priors[age, time].prior)
                else:
                    row["value_prior_id"] = None
                if smooth.d_age_priors:
                    row["dage_prior_id"] = prior_objects.index(smooth.d_age_priors[age, time].prior)
                else:
                    row["dage_prior_id"] = None
                if smooth.d_time_priors:
                    row["dtime_prior_id"] = prior_objects.index(smooth.d_time_priors[age, time].prior)
                else:
                    row["dtime_prior_id"] = None
                rows.append(row)

    return pd.DataFrame(
        rows, columns=["age", "time", "const_value", "value_prior_id", "dage_prior_id", "dtime_prior_id"]
    )


def _smooth_row(name, smooth, grid, prior_objects):
    if smooth.value_priors and smooth.value_priors.hyper_prior:
        mulstd_value_prior_id = prior_objects.index(smooth.value_priors.hyper_prior)
    else:
        mulstd_value_prior_id = np.nan
    if smooth.d_age_priors and smooth.d_age_priors.hyper_prior:
        mulstd_dage_prior_id = prior_objects.index(smooth.d_age_priors.hyper_prior)
    else:
        mulstd_dage_prior_id = np.nan
    if smooth.d_time_priors and smooth.d_time_priors.hyper_prior:
        mulstd_dtime_prior_id = prior_objects.index(smooth.d_time_priors.hyper_prior)
    else:
        mulstd_dtime_prior_id = np.nan

    return {
        "smooth_name": name,
        "n_age": len(grid.age.unique()),
        "n_time": len(grid.time.unique()),
        "mulstd_value_prior_id": mulstd_value_prior_id,
        "mulstd_dage_prior_id": mulstd_dage_prior_id,
        "mulstd_dtime_prior_id": mulstd_dtime_prior_id,
    }


def make_smooth_and_smooth_grid_tables(context, age_table, time_table, prior_objects):
    grid_tables = []
    smooths = []
    for rate in context.rates:
        def process_smooth(smooth, smooth_type):
            grid_table = make_smooth_grid_table(smooth, prior_objects)
            smooths.append(_smooth_row(f"{rate.name}_{smooth_type}_smooth", smooth, grid_table, prior_objects))
            grid_table["smooth_id"] = len(smooths)
            grid_tables.append(grid_table)
        if rate.parent_smooth:
            process_smooth(rate.parent_smooth, "parent")
        for smooth in rate.child_smoothings:
            process_smooth(smooth, "child")

    if grid_tables:
        grid_table = pd.concat(grid_tables)
        grid_table["smooth_grid_id"] = grid_table.index
        grid_table = pd.merge_asof(grid_table.sort_values("age"), age_table, on="age").drop("age", "columns")
        grid_table = pd.merge_asof(grid_table.sort_values("time"), time_table, on="time").drop("time", "columns")
    else:
        grid_table = pd.DataFrame()
    smooth_table = pd.DataFrame(smooths)
    smooth_table["smooth_id"] = smooth_table.index

    return smooth_table, grid_table
