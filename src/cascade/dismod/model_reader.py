from copy import deepcopy
from math import isnan

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.dismod.constants import IntegrandEnum, RateEnum
from cascade.dismod.dismod_groups import DismodGroups
from cascade.dismod.var import Var

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


def read_vars(dismod_file, var_ids, which):
    """

    Args:
        dismod_file:
        var_ids (DismodGroups): The output of ``read_var_table_as_id``.
        which (str): Could be "start_var", "truth_var", "scale_var", "fit_var".

    Returns:
        DismodGroups: With means for everything.
    """
    var_name = f"{which}_var"
    table = getattr(dismod_file, var_name)
    if table.empty:
        raise AttributeError(f"Dismod file has no data in {var_name} table.")
    vars = DismodGroups()
    for group_name, group in var_ids.items():
        for key, value in group.items():
            vars[group_name][key] = _read_vars_one_field(table, var_name, value)
    return vars


def _construct_vars_one_field(name, var_id, new_var):
    id_column = f"{name}_id"
    var_column = f"{name}_value"
    with_id = new_var.grid.merge(var_id.grid[["age", "time", "var_id"]], on=["age", "time"], how="left")
    return pd.DataFrame({
        id_column: with_id.var_id.values,
        var_column: with_id["mean"].values,
    })


def _read_vars_one_field(table, name, id_draw):
    id_column = f"{name}_id"
    var_column = f"{name}_value"
    vals = deepcopy(id_draw)
    table = table.reset_index(drop=True)
    with_var = vals.grid.merge(table, left_on="var_id", right_on=id_column, how="left")
    with_var = with_var.drop(columns=["var_id"])
    vals.grid = with_var.rename(columns={var_column: "mean"})

    for mulstd, mul_id in vals.mulstd.items():
        vals.mulstd[mulstd] = float(table.query("@id_column == @mul_id")[var_column])
    return vals


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
    smooths = read_smooths(dismod_file, child_node)

    # The inverted smooth is a map from the smooth id back to the DismodGroup.
    inverted_smooth = dict()
    for group_name, group in smooths.items():
        for key, smooth_value in group.items():
            if group_name == "random_effect":
                child_node = key[1]
                inverted_smooth[(smooth_value, child_node)] = (group_name, key)
            else:
                inverted_smooth[(smooth_value, parent_node)] = (group_name, key)

    # This groupby to separate the smooth grids is split into two levels because
    # the second level is node_id, which is nan for mulcovs, and groupby
    # excludes rows with nans in the keys.
    var_ids = DismodGroups()
    age, time = (dismod_file.age, dismod_file.time)
    for smooth_id, sub_grid_df in dismod_file.var.groupby(["smooth_id"]):
        if sub_grid_df[sub_grid_df.var_type == "rate"].empty:
            _add_one_field_to_vars(inverted_smooth, parent_node, smooth_id, sub_grid_df, var_ids, age, time)
        else:
            # Multiple random effects, identified as "rates," can share a smooth.
            for node_id, re_grid_df in sub_grid_df.groupby(["node_id"]):
                _add_one_field_to_vars(inverted_smooth, node_id, smooth_id, re_grid_df, var_ids, age, time)

    if var_ids.count() != len(dismod_file.var):
        MATHLOG.error(f"Found {var_ids.count()} of {len(dismod_file.var)} vars in db file.")
    return var_ids


def _add_one_field_to_vars(inverted_smooth, node_id, smooth_id, sub_grid_df, var_ids, age, time):
    at_grid_df = sub_grid_df[sub_grid_df.age_id.notna() & sub_grid_df.time_id.notna()]
    at_grid_df = at_grid_df.merge(age, on="age_id", how="left") \
        .merge(time, on="time_id", how="left") \
        .drop(columns=["age_id", "time_id"])
    draw = Var((np.unique(at_grid_df.age.values), np.unique(at_grid_df.time.values)))
    draw.grid = draw.grid.merge(
        at_grid_df[["age", "time", "var_id"]], how="left", on=["age", "time"])
    # The mulstd hyper-priors aren't indexed by age and time, so separate.
    for kind in ["value", "dage", "dtime"]:
        match = sub_grid_df.query("var_type == @kind")
        if not match.empty:
            draw.mulstd[kind] = match.var_id
    group_name, key = inverted_smooth[(smooth_id, node_id)]
    var_ids[group_name][key] = draw


def convert_age_time_to_values(dismod_file, draw_parts):
    for group in draw_parts.values():
        for draw in group.values():
            draw.values = draw.grid.merge(dismod_file.age, on="age_id", how="left") \
                .merge(dismod_file.time, on="time_id", how="left") \
                .drop(columns=["age_id", "time_id"])
            draw.ages = np.sort(np.unique(draw.grid.age.values))
            draw.times = np.sort(np.unique(draw.grid.time.values))


def read_smooths(dismod_file, child_node):
    """Construct a DismodGroups where the value is the ID of the smooth table
    for that group. This will be very helpful for interpreting the var table."""
    smooths = DismodGroups()
    _read_rate_smooths(child_node, dismod_file.nslist_pair, dismod_file.rate, smooths)
    _read_mulcov_smooths(dismod_file.mulcov, dismod_file.covariate, smooths)
    return smooths


def _read_rate_smooths(child_node, nslist_pair_table, rate_table, smooths):
    for rate_row in rate_table.itertuples():
        if not isnan(rate_row.parent_smooth_id):
            smooths.rate[rate_row.rate_name] = int(rate_row.parent_smooth_id)
        if not isnan(rate_row.child_smooth_id):
            # Random effects can have children with different fields but same smoothing.
            for child in child_node:
                smooths.random_effect[(rate_row.rate_name, child)] = int(rate_row.child_smooth_id)
        if not isnan(rate_row.child_nslist_id):
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
        covariate_name = str(found_name.iloc[0].c_covariate_name)
        if mulcov_row.mulcov_type == "rate_value":
            target_name = RateEnum(mulcov_row.rate_id).name
        else:
            target_name = IntegrandEnum(mulcov_row.integrand_id).name

        MATHLOG.debug(f"Covariate={covariate_name}, target={target_name}")
        group[mulcov_row.mulcov_type][(covariate_name, target_name)] = mulcov_row.smooth_id


def read_parent_node(dismod_file):
    return int(dismod_file.option.query("option_name == 'parent_node_id'").option_value)


def read_child_nodes(dismod_file, parent_node):
    return dismod_file.node[dismod_file.node.parent == parent_node].node_id.values
