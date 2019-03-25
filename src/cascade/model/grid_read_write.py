from functools import partial
from math import isnan, nan

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.constants import IntegrandEnum, RateEnum, PriorKindEnum
from cascade.model.age_time_grid import AgeTimeGrid
from cascade.model.dismod_groups import DismodGroups
from cascade.model.smooth_grid import SmoothGrid
from cascade.model.var import Var

CODELOG, MATHLOG = getLoggers(__name__)


def write_vars(dismod_file, new_vars, var_ids, which):
    """

    Args:
        dismod_file:
        new_vars (DismodGroups): The new vars to write.
        var_ids (DismodGroups): The output of ``read_var_table_as_id``.
        which (str): Could be "start_var", "truth_var", "scale_var", "fit_var".
    """
    var_name = f"{which}_var"
    id_column = f"{var_name}_id"
    total = list()
    for group_name, group in var_ids.items():
        for key, value in group.items():
            total.append(_construct_vars_one_field(var_name, value, new_vars[group_name][key]))
    new_table = pd.concat(total).sort_values(by=[id_column])
    setattr(dismod_file, var_name, new_table)


def _construct_vars_one_field(name, var_id, new_var):
    id_column = f"{name}_id"
    var_column = f"{name}_value"
    with_id = new_var.grid.merge(var_id.grid[["age", "time", "var_id"]], on=["age", "time"], how="left")
    if with_id.var_id.isna().any():
        na_vals = with_id[with_id.var_id.isna()]
        raise RuntimeError(f"Could not align {id_column} with var_id. {na_vals} when writing to db file.")
    return pd.DataFrame({
        id_column: with_id.var_id.values,
        var_column: with_id["mean"].values,
    })


def read_vars(dismod_file, var_ids, which):
    """
    Reads a table full of vars (start, truth, scale).

    Args:
        dismod_file:
        var_ids (DismodGroups): The output of ``read_var_table_as_id``.
        which (str): Could be "start_var", "truth_var", "scale_var", "fit_var".

    Returns:
        DismodGroups: With means for everything.
    """
    var_name = f"{which}_var"
    table = getattr(dismod_file, var_name)
    return _assign_from_var_ids(table, var_ids, partial(_read_vars_one_field, name=var_name))


def _assign_from_var_ids(table, var_ids, var_builder):
    """Iterate over var ids and execute a function on each grid in the varids.
    The ``var_builder`` creates a Var table from a table and a set of var_ids
    to read from that table.
    """
    if table.empty:
        raise AttributeError(f"Dismod file has no data in table {table.columns} during read from vars.")
    var_groups = DismodGroups()
    for group_name, group in var_ids.items():
        for key, var_id_mapping in group.items():
            var_groups[group_name][key] = var_builder(table, var_id_mapping)
    return var_groups


def _read_vars_one_field(table, id_draw, name):
    """Read one rate or random effect's values.

    Args:
        table: Dismod-AT table represented by a Pandas dataframe.
        id_draw (AgeTimeGrid): An AgeTimeGrid that contains rows of ["var_id"].
        name (str): name of the table, following Dismod-AT conventions.

    Returns:
        A new Var containing the values at these ages and times.
    """
    id_column = f"{name}_id"
    var_column = f"{name}_value"
    table = table.reset_index(drop=True)
    with_var = id_draw.grid.merge(table, left_on="var_id", right_on=id_column, how="left")

    vals = Var(id_draw.ages, id_draw.times)
    vals.grid = vals.grid.assign(age=with_var["age"], time=with_var["time"], mean=with_var[var_column])

    for mulstd, mul_id in id_draw.mulstd.items():
        if mul_id.var_id.notna().all():
            mulstd_id = int(mul_id.var_id.iloc[0])
            row = table.loc[table[id_column] == mulstd_id, var_column]
            vals.set_mulstd("value", float(row))
    return vals


def read_prior_residuals(dismod_file, var_ids):
    """Read residuals on value, dage, and dtime priors. Includes lagrange values."""
    return _assign_from_var_ids(dismod_file.fit_var, var_ids, _read_residuals_one_field)


def _read_residuals_one_field(table, id_draw):
    """Reads residuals for the Priors."""
    # Get the data out.
    table = table.reset_index(drop=True)
    with_var = id_draw.grid.merge(table, left_on="var_id", right_on="fit_var_id", how="left")

    # Set up the container
    residual = [f"residual_{pk.name}" for pk in PriorKindEnum]
    lagrange = [f"lagrange_{pk.name}" for pk in PriorKindEnum]
    data_cols = ["fit_var_value"] + residual + lagrange
    vals = AgeTimeGrid(id_draw.ages, id_draw.times, columns=data_cols)

    # Fill the container
    vals.grid = vals.grid.drop(columns=data_cols) \
        .merge(with_var[["age", "time"] + data_cols])

    for mulstd, mul_id in id_draw.mulstd.items():
        if mul_id.var_id.notna().all():
            mulstd_id = int(mul_id.var_id.iloc[0])  # noqa: F841
            row = table.loc[table["fit_var_id"] == mulstd_id, data_cols]
            vals.mulstd[mulstd].loc[:, data_cols] = row.values

    return vals


def read_samples(dismod_file, var_ids):
    """Get output of Dismod-AT sample command."""
    return _assign_from_var_ids(dismod_file.sample, var_ids, _samples_one_field)


def _samples_one_field(table, id_draw):
    # Get the data out.
    table = table.reset_index(drop=True)
    with_var = table.merge(id_draw.grid, left_on="var_id", right_on="var_id", how="inner")

    # This is an AgeTimeGrid container, with multiple samples.
    # It will use the idx column to represent the sample index.
    vals = AgeTimeGrid(id_draw.ages, id_draw.times, columns=["mean", "idx"])
    vals.grid = with_var[["age", "time", "var_value", "sample_index"]].rename(
        columns={"var_value": "mean", "sample_index": "idx"})

    for mulstd, mul_id in id_draw.mulstd.items():
        if mul_id.var_id.notna().all():
            mulstd_id = int(mul_id.var_id.iloc[0])  # noqa: F841
            vals.mulstd[mulstd] = table[table.var_id == mulstd_id][["sample_index", "var_value"]] \
                .rename(columns={"sample_index": "idx", "var_value": "mean"}) \
                .assign(age=nan, time=nan)

    return vals


def read_simulation_model(dismod_file, original_model, var_ids, index):
    """After simulate was run, it makes a new model. This takes
    an existing model and modifies its priors so that we can run again."""
    sim_priors = dismod_file.prior_sim[dismod_file.prior_sim.simulate_index == index]
    sim_model = original_model.model_like()
    for group_name, group in var_ids.items():
        for var_key, var_grid in group.items():
            use_key = var_key
            try:
                model_grid = original_model[group_name][var_key]
            except KeyError:
                # This handles the case that a random effect has one smooth grid
                # versus a random effect having a smooth grid per child.
                if (var_key[0], None) in original_model[group_name]:
                    use_key = (var_key[0], None)
                    model_grid = original_model[group_name][use_key]
                else:
                    raise
            constructed_prior_grid = SmoothGrid(var_grid.ages, var_grid.times)
            _read_one_prior_sim_grid(model_grid, constructed_prior_grid, sim_priors, var_grid)
            _read_one_prior_sim_mulstd(model_grid, constructed_prior_grid, sim_priors, var_grid)
            sim_model[group_name][use_key] = constructed_prior_grid
    return sim_model


def _read_one_prior_sim_grid(model_grid, priors, sim_priors_df, var_grid):
    # For values in the age-time grid, there are three prior types for
    # each value of the grid.
    for age, time in var_grid.age_time():
        float_var_id = var_grid[age, time].var_id
        if not isnan(float_var_id):
            in_priors = sim_priors_df.var_id == int(float_var_id)
        else:
            in_priors = None
        if in_priors is not None and in_priors.any():
            for kind in ["value", "dage", "dtime"]:
                prior_mean = float(sim_priors_df[in_priors][f"prior_sim_{kind}"])
                dest_priors = getattr(priors, kind)
                source_prior = getattr(model_grid, kind)[age, time]
                if not isnan(prior_mean) and source_prior is not None:
                    dest_priors[age, time] = source_prior.assign(mean=prior_mean)
                elif source_prior is not None:
                    dest_priors[age, time] = source_prior
                # else no prior to set because dage and dtime priors at endpoints
                # aren't required.
        else:
            for kind in ["value", "dage", "dtime"]:
                getattr(priors, kind)[age, time] = getattr(model_grid, kind)[age, time]


def _read_one_prior_sim_mulstd(model_grid, priors, sim_priors_df, var_grid):
    # For the mulstd, there are three different var_ids, one for each kind.
    for mulstd_kind in ["value", "dage", "dtime"]:
        mulstd_var_id = float(var_grid.mulstd[mulstd_kind]["var_id"])
        source_prior = getattr(model_grid, mulstd_kind).mulstd_prior
        if not isnan(mulstd_var_id):
            prior_mean = float(sim_priors_df[sim_priors_df.var_id == mulstd_var_id][f"prior_sim_{mulstd_kind}"])
            if not isnan(prior_mean):
                getattr(priors, mulstd_kind).mulstd_prior = source_prior.assign(mean=prior_mean)
            else:
                getattr(priors, mulstd_kind).mulstd_prior = source_prior
        else:
            getattr(priors, mulstd_kind).mulstd_prior = source_prior


def read_var_table_as_id(dismod_file):
    """
    This reads the var table in order to find the ids for all of the vars.
    It puts those into a DismodGroups which can then decode any table
    associated with the var table.
    The empty vars come from the Model.model_variables property.
    This ``var_id`` table has ``age`` and ``time`` and uses real
    locations instead of ``node_id``. It's meant for joins with tables
    in the Dismod file.
    """
    parent_node = read_parent_node(dismod_file)
    child_node = read_child_nodes(dismod_file, read_parent_node(dismod_file))

    # The var table has a random field for each smooth_id, except for the
    # random effects. It doesn't identify which random field each smooth id
    # goes with, so we first make the DismodGroups structure of smooths
    # and then invert it in order to read the vars table.
    inverted_smooth = read_inverted_smooths(dismod_file, parent_node, child_node)

    # This groupby to separate the smooth grids is split into two levels because
    # the second level is node_id, which is nan for mulcovs, and groupby
    # excludes rows with nans in the keys.
    var_ids = DismodGroups()
    age, time = (dismod_file.age, dismod_file.time)
    for smooth_id, sub_grid_df in dismod_file.var.groupby(["smooth_id"]):
        if sub_grid_df[sub_grid_df.var_type == "rate"].empty:
            group_name, key = inverted_smooth[(smooth_id, parent_node)]
            var_ids[group_name][key] = _construct_var_id_from_var_table(sub_grid_df, age, time)
        else:
            # Multiple random effects, identified as "rates," can share a smooth.
            for node_id, re_grid_df in sub_grid_df.groupby(["node_id"]):
                group_name, key = inverted_smooth[(smooth_id, node_id)]
                var_ids[group_name][key] = _construct_var_id_from_var_table(re_grid_df, age, time)

    if var_ids.variable_count() != len(dismod_file.var):
        MATHLOG.error(f"Found {var_ids.variable_count()} of {len(dismod_file.var)} vars in db file.")

    return rename_node_to_location(dismod_file.node, var_ids)


def _construct_var_id_from_var_table(sub_grid_df, age, time):
    at_grid_df = sub_grid_df[sub_grid_df.age_id.notna() & sub_grid_df.time_id.notna()]
    if age.index.name != "age_id":
        age = age.set_index("age_id")
    if time.index.name != "time_id":
        time = time.set_index("time_id")
    at_grid_df = at_grid_df.merge(age, left_on="age_id", right_index=True, how="left") \
        .merge(time, left_on="time_id", right_index=True, how="left")
    draw = AgeTimeGrid(np.unique(at_grid_df.age.values), np.unique(at_grid_df.time.values), ["var_id"])
    draw.grid = draw.grid.drop(columns=["var_id"]) \
        .merge(at_grid_df[["age", "time", "var_id"]], how="left", on=["age", "time"])
    # The mulstd hyper-priors aren't indexed by age and time, so separate.
    for kind in ["value", "dage", "dtime"]:
        mulstd_name = f"mulstd_{kind}"  # noqa: F841
        match = sub_grid_df.query("var_type == @mulstd_name")
        if not match.empty:
            draw.mulstd[kind].at[0, "var_id"] = match.var_id
    return draw


def read_inverted_smooths(dismod_file, parent_node, child_node):
    """Construct a DismodGroups where the value is the ID of the smooth table
    for that group. This will be very helpful for interpreting the var table."""
    smooths = DismodGroups()
    _read_rate_smooths(child_node, dismod_file.nslist_pair, dismod_file.rate, smooths)
    _read_mulcov_smooths(dismod_file.mulcov, dismod_file.covariate, smooths)

    # The inverted smooth is a map from the smooth id back to the DismodGroup.
    inverted_smooth = dict()
    for group_name, group in smooths.items():
        for key, smooth_value in group.items():
            if group_name == "random_effect":
                child_node = key[1]
                inverted_smooth[(smooth_value, child_node)] = (group_name, key)
            else:
                inverted_smooth[(smooth_value, parent_node)] = (group_name, key)

    return inverted_smooth


def _is_a_smooth(smooth_id):
    return smooth_id is not None and not isnan(smooth_id)


def _read_rate_smooths(child_node, nslist_pair_table, rate_table, smooths):
    for rate_row in rate_table.itertuples():
        if _is_a_smooth(rate_row.parent_smooth_id):
            smooths.rate[rate_row.rate_name] = int(rate_row.parent_smooth_id)
        if _is_a_smooth(rate_row.child_smooth_id):
            # Random effects can have children with different fields but same smoothing.
            for child in child_node:
                smooths.random_effect[(rate_row.rate_name, child)] = int(rate_row.child_smooth_id)
        if _is_a_smooth(rate_row.child_nslist_id):
            child_df = nslist_pair_table[nslist_pair_table.nslist_id == rate_row.child_nslist_id]
            for ns_row in child_df.itertuples():
                smooths.random_effect[(rate_row.rate_name, ns_row.node_id)] = int(ns_row.smooth_id)


def _read_mulcov_smooths(mulcov_table, covariate_table, smooths):
    group = dict(rate_value=smooths.alpha, meas_value=smooths.beta, meas_std=smooths.gamma)
    for mulcov_row in mulcov_table.itertuples():
        found_name = covariate_table.query("covariate_id == @mulcov_row.covariate_id")
        if found_name.empty:
            MATHLOG.error(f"Mulcov covariate id {mulcov_row.covariate_id} not found in covariate table.")
            raise RuntimeError(f"Could not find covariate id {mulcov_row.covariate_id} in covariate table.")
        covariate_name = str(found_name.iloc[0].covariate_name)
        if mulcov_row.mulcov_type == "rate_value":
            target_name = RateEnum(mulcov_row.rate_id).name
        else:
            target_name = IntegrandEnum(mulcov_row.integrand_id).name

        MATHLOG.debug(f"Covariate={covariate_name}, target={target_name}")
        group[mulcov_row.mulcov_type][(covariate_name, target_name)] = mulcov_row.smooth_id


def read_parent_node(dismod_file):
    """Get ``node_id`` for parent location."""
    return int(dismod_file.option.query("option_name == 'parent_node_id'").option_value)


def read_child_nodes(dismod_file, parent_node):
    """Get node ids for child locations."""
    return dismod_file.node[dismod_file.node.parent == parent_node].node_id.values


def rename_node_to_location(node_table, var_ids):
    """Given a DismodGroups based on node_id, change keys
    to be based on location_id"""
    node_to_location = DismodGroups()
    for group_name, group in var_ids.items():
        for key, smooth_value in group.items():
            if group_name == "random_effect":
                this_node = key[1]  # noqa: F841
                child_location = int(node_table.query("node_id == @this_node").c_location_id)
                node_to_location[group_name][(key[0], child_location)] = smooth_value
            else:
                node_to_location[group_name][key] = smooth_value

    return node_to_location
