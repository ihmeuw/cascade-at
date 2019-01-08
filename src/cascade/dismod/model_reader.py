from copy import deepcopy
from math import isnan
import numpy as np

from cascade.core.log import getLoggers
from cascade.model.random_field import FieldDraw, PartsContainer

CODELOG, MATHLOG = getLoggers(__name__)


def read_vars(dismod_file, var_ids, which):
    """

    Args:
        dismod_file:
        var_ids (PartsContainer): The output of ``read_var_table_as_id``.
        which (str): Could be "start_var", "truth_var", "scale_var", "fit_var".

    Returns:
        PartsContainer: With means for everything.
    """
    table = getattr(dismod_file, which)
    return PartsContainer.fromtuples((k, _read_vars_one_field(table, which, v)) for (k, v) in var_ids.items())


def _read_vars_one_field(table, name, id_draw):
    id_column = f"{name}_id"
    var_column = f"{name}_id"
    vals = deepcopy(id_draw)
    vals.values = vals.values.merge(table, left_on="var_id", right_on=id_column, how="left") \
        .drop("var_id") \
        .rename(columns={var_column, "mean"})
    for mulstd, mul_id in vals.mulstd.items():
        vals.mulstd[mulstd] = float(table.query("@id_column == @mul_id")[var_column])


def read_var_table_as_id(dismod_file):
    """
    This reads the var table in order to find the ids for all of the vars.
    It puts thos into a PartsContainer which can then decode any table
    associated with the var table.
    The empty vars come from the Model.model_variables property.
    """
    parent_node = read_parent_node(dismod_file)
    child_node = read_child_nodes(dismod_file, read_parent_node(dismod_file))

    # The var table has a random field for each smooth_id, except for the
    # random effects. It doesn't identify which random field each smooth id
    # goes with, so we first make the PartsContainer structure of smooths
    # and then invert it in order to read the vars table.
    smooths = read_smooths(dismod_file, child_node)

    inverted_smooth = dict()
    for parts_key, smooth_invert in smooths.items():
        if parts_key[0] == "random_effect":
            child_node = parts_key[2]
            inverted_smooth[(smooth_invert, child_node)] = parts_key
        else:
            inverted_smooth[(smooth_invert, parent_node)] = parts_key

    var_ids = PartsContainer()
    for (smooth_id, node_id), sub_grid_df in dismod_file.var.groupby(["smooth_id", "node_id"]):
        part, *rest = inverted_smooth[(smooth_id, node_id)]
        at_grid_df = sub_grid_df[sub_grid_df.age_id.notna() & sub_grid_df.time_id.notna()]
        age_ids = np.unique(at_grid_df.age_id.values)
        time_ids = np.unique(at_grid_df.time_id.values)
        draw = FieldDraw((age_ids, time_ids))
        id_df = draw.values.merge(at_grid_df[["age_id", "time_id", "var_id"]], how="left", on=["age_id", "time_id"])
        draw.values = id_df

        for kind in ["value", "dage", "dtime"]:
            match = sub_grid_df.query("var_type == @kind")
            if not match.empty:
                draw.mulstd[kind] = match.var_id

        setattr(var_ids, part, draw)

    convert_age_time_to_values(dismod_file, var_ids)
    if len(var_ids) != len(dismod_file.var):
        MATHLOG.error(f"Found {len(var_ids)} of {len(dismod_file.var)} vars in db file.")
    return var_ids


def convert_age_time_to_values(dismod_file, draw_parts):
    for draw in draw_parts.values():
        draw.values = draw.values.merge(dismod_file.ages, on="age_id", how="left") \
            .merge(dismod_file.times, on="time_id", how="left") \
            .drop(columns=["age_id", "time_id"])
        draw.ages = np.sort(np.unique(draw.values.age.values))
        draw.times = np.sort(np.unique(draw.values.time.values))


def read_smooths(dismod_file, child_node):
    smooths = PartsContainer()
    for rate_row in dismod_file.rate.itertuples():
        if not isnan(rate_row.parent_smooth_id):
            smooths.rate[rate_row.rate_name] = rate_row.parent_smooth_id
        if not isnan(rate_row.child_smooth_id):
            # Random effects can have children with different fields but same smoothing.
            for child in child_node:
                smooths.random_effect[(rate_row.rate_name, child)] = rate_row.child_smooth_id
        if not isnan(rate_row.child_nslist_id):
            child_df = dismod_file.nslist_pair_id[dismod_file.nslist_pair_id.nslist_id == rate_row.child_nslist_id]
            for ns_row in child_df.itertuples():
                smooths.random_effect[(rate_row.rate_name, ns_row.node_id)] = ns_row.smooth_id
    for mulcov_row in dismod_file.mulcov.itertuples():
        if mulcov_row.mulcov_type == "rate_value":
            smooths.alpha[(mulcov_row.covariate_id, mulcov_row.rate_id)] = mulcov_row.smooth_id
        elif mulcov_row.mulcov_type == "meas_value":
            smooths.beta[(mulcov_row.covariate_id, mulcov_row.integrand_id)] = mulcov_row.smooth_id
        elif mulcov_row.mulcov_type == "meas_std":
            smooths.gamma[(mulcov_row.covariate_id, mulcov_row.integrand_id)] = mulcov_row.smooth_id
        else:
            raise RuntimeError(f"Unknown mulcov type {mulcov_row.mulcov_type}")
    return smooths


def read_parent_node(dismod_file):
    return int(dismod_file.options.query("option_name == 'parent_node_id'").option_value)


def read_child_nodes(dismod_file, parent_node):
    return dismod_file.node_id[dismod_file.node_id.parent == parent_node].node_id.values
