from math import nan

import numpy as np

from cascade.core import getLoggers
from cascade.dismod.constants import DensityEnum, INTEGRAND_TO_WEIGHT, IntegrandEnum

CODELOG, MATHLOG = getLoggers(__name__)


def read_data_residuals(dismod_file):
    """Reads residuals indexed by the name of the data line.

    Args:lagrange_dtime
        dismod_file: The DismodFile wrapper.

    Returns:
        DataFrame: Columns are ``name``, ``avg_integrand``,
        ``weighted_residual``.
    """
    # The data table associates data_id with data_name, but data_subset
    # has an id for all data that were used and will have residuals.
    data_subset = dismod_file.data_subset.reset_index(drop=True)
    subset_id_to_name = data_subset.merge(dismod_file.data[["data_id", "data_name"]], on="data_id") \
        .drop(columns=["data_id"])
    return dismod_file.fit_data_subset.reset_index(drop=True).merge(
        subset_id_to_name, left_on="fit_data_subset_id", right_on="data_subset_id") \
        .drop(columns=["data_subset_id", "fit_data_subset_id"]) \
        .rename(columns={"data_name": "name"})


def write_data(dismod_file, data, covariate_rename):
    """
    Writes a data table. Locations can be any location and will be pruned
    to those that are descendants of the parent location.

    Args:
        dismod_file: Wrapper for a dismod db file.
        data (pd.DataFrame): Columns are ``integrand``, ``location``,
            ``name``, ``hold_out``,
            ``age_lower``, ``age_upper``, ``time_lower``, ``time_upper``,
            ``density``, ``mean``, ``std``, ``eta``, ``nu``.
            The ``name`` is optional and will be assigned from the index.
            In addition, covariate columns are included. If ``hold_out``
            is missing, it will be assigned ``hold_out=0`` for not held out.
            If nu or eta aren't there, they will be added. If ages
            or times are listed as ``age`` and ``time``, they will be
            considered point values and expanded into upper and lower.
        covariate_rename (Dict[str,str]): Map from user names to Dismod names
            for covariates. Dismod needs renaming of covariates to ``x_0``,
             ``x_1``, etc.
    """
    if data is None or data.empty:
        dismod_file.data = dismod_file.empty_table("data")
        return

    required_columns = ["location", "integrand", "density", "mean", "std"]
    missing_columns = ', '.join(str(mc) for mc in (set(required_columns) - set(data.columns)))
    if missing_columns:
        have_columns = ', '.join(str(hc) for hc in data.columns)
        raise ValueError(f"Data is missing columns {missing_columns}. Have {have_columns}")
    # Some of the columns get the same treatment as the average integrand.
    # This takes care of integrand and location.
    like_avgint = avgint_to_dataframe(dismod_file, data, covariate_rename).drop(columns=["avgint_id"])
    # Other columns have to do with priors.
    with_density = like_avgint.assign(density_id=like_avgint.density.apply(lambda x: DensityEnum[x].value))
    _check_column_assigned(with_density, "density")
    with_density = with_density.reset_index(drop=True).drop(columns=["density"])
    dismod_file.data = with_density.rename(
        columns={"mean": "meas_value", "std": "meas_std", "name": "data_name"})


def _add_data_age_time_range(dismod_file, data):
    """
    Add to the age and time tables the extremal values of data.

    Dismod-AT uses the age and time table minimum and maximum to determine
    the extent of integration. If the data exceeds the bounds of priors
    that are already registered in the age and time table, then you have to
    add min and max values to those tables.
    """
    if data is not None and not data.empty:
        for dimension in ["age", "time"]:
            cols = [ac for ac in data.columns if ac.startswith(dimension)]
            if not cols:
                raise ValueError(f"Dataframe must have age and time columns but has {data.columns}.")
            dm_table = getattr(dismod_file, dimension)
            assert f"{dimension}_id" in dm_table.columns
            small = data[cols].min().min()
            if small < dm_table[dimension].min():
                dm_table = dm_table.append(
                    {f"{dimension}_id": len(dm_table), dimension: small}, ignore_index=True)
            large = data[cols].max().max()
            if large > dm_table[dimension].max():
                dm_table = dm_table.append(
                    {f"{dimension}_id": len(dm_table), dimension: large}, ignore_index=True)
            setattr(dismod_file, dimension, dm_table)


def avgint_to_dataframe(dismod_file, avgint, covariate_rename):
    """
    Translate integrand name to id. Translate location to node.
    Add weight appropriate for this integrand. Writes to the Dismod file.

    Args:
        dismod_file: Wrapper class for Dismod db file.
        avgint (pd.DataFrame): Columns are ``integrand``, ``location``,
            ``age_lower``, ``age_upper``, ``time_lower``, ``time_upper``.
        covariate_rename (Dict[str,str]): Covariates are renamed for underlying
            db file. This is from user names to db file names.
    """
    expect_columns = {"integrand", "location", "age_lower", "age_upper",
                      "time_lower", "time_upper"}
    if expect_columns - set(avgint.columns):
        raise KeyError(f"Expect avgint dataframe to have columns {expect_columns} "
                       f"not {avgint.columns}")
    avgint = avgint.reset_index(drop=True)
    # Create the avgint_id early so we can write in the same order it was given.
    with_id = avgint.assign(avgint_id=avgint.index)
    with_id = with_id.assign(integrand_id=with_id.integrand.apply(lambda x: IntegrandEnum[x].value))
    _check_column_assigned(with_id, "integrand")
    with_weight = with_id.assign(weight_id=with_id.integrand.apply(lambda x: INTEGRAND_TO_WEIGHT[x].value))
    with_weight = with_weight.drop(columns=["integrand"]).reset_index(drop=True)
    with_weight = with_weight.assign(location=with_weight.location.astype(np.int))
    with_location = with_weight.merge(
        dismod_file.node[["c_location_id", "node_id"]], left_on="location", right_on="c_location_id", how="left") \
        .drop(columns=["c_location_id", "location"])
    _add_data_age_time_range(dismod_file, avgint)
    with_location = with_location.rename(columns=covariate_rename)
    return with_location.reset_index(drop=True)


def read_avgint(dismod_file):
    """Read average integrand cases, translating to locations and covariates."""
    avgint = dismod_file.avgint
    with_integrand = avgint.assign(integrand=avgint.integrand_id.apply(lambda x: IntegrandEnum(x).name))
    with_location = with_integrand.merge(dismod_file.node, on="node_id", how="left") \
        .rename(columns={"c_location_id": "location"})
    usual_columns = ["avgint_id", "location", "integrand", "age_lower", "age_upper",
                     "time_lower", "time_upper"]
    covariate_map = _dataframe_as_dict(dismod_file.covariate, "covariate_id", "covariate_name")
    covariate_rename = {cc: covariate_map[int(cc.lstrip("x_"))] for cc in with_location.columns if cc.startswith("x_")}
    with_covariates = with_location.rename(columns=covariate_rename)
    return with_covariates[usual_columns + list(covariate_rename.values())]


def _dataframe_as_dict(df, key_column, value_column):
    """Given DataFrame(id=[1,2,3], name=['a', 'b', 'c']), this gives
    {1: 'a', 2: 'b', 3: 'c'}.
    """
    return {getattr(row, key_column): getattr(row, value_column) for row in df.itertuples()}


def _check_column_assigned(with_id, column):
    column_id = f"{column}_id"
    unassigned_rows = with_id[with_id[column_id].isna()]
    if np.any(unassigned_rows):
        not_found_integrand = with_id[unassigned_rows][column].unique()
        kind_enum = globals()[f"{column.capitalize()}Enum"]
        err_message = (f"The {column} {not_found_integrand} weren't found in the "
                       f"{column} list {[i.name for i in kind_enum]}.")
        MATHLOG.error(err_message)
        raise RuntimeError(err_message)


def read_simulation_data(dismod_file, data, index):
    """After simulate was run, it makes new data. This takes an existing
    set of data and modifies its values with the simulated version so that
    we can fit again.

    The data has been subset into the data_subset, and then simulate indexes
    into that data subset. This rebuilds back to the original data.
    """
    # The saved dataset links the unique name to the data_id, but use the data
    # passed in for the rest.
    db_data = dismod_file.data
    # Links data_subset_id to data_id.
    data_subset = dismod_file.data_subset.reset_index(drop=True)
    data_sim = dismod_file.data_sim  # The actual answer.

    keep_sim_columns = ["data_subset_id", "data_sim_value", "data_sim_delta"]
    index_subset = data_sim.loc[data_sim.simulate_index == index, keep_sim_columns]

    aligned = index_subset.merge(data_subset, on="data_subset_id", how="left") \
        .merge(db_data[["data_id", "data_name"]], on="data_id") \
        .drop(columns=["data_subset_id", "data_id"])
    augmented = data.merge(aligned, left_on="name", right_on="data_name", how="left")
    augmented = augmented.drop(columns="data_name")
    augmented.loc[augmented.data_sim_value.notna(), "mean"] = augmented.data_sim_value
    augmented.loc[augmented.data_sim_delta.notna(), "std"] = augmented.data_sim_delta
    return augmented.drop(columns=["data_sim_value", "data_sim_delta"])


def point_age_time_to_interval(data):
    if data is None:
        return
    for at in ["age", "time"]:  # Convert from point ages and times.
        for lu in ["lower", "upper"]:
            if f"{at}_{lu}" not in data.columns and at in data.columns:
                data = data.assign(**{f"{at}_{lu}": data[at]})
    return data.drop(columns={"age", "time"} & set(data.columns))


def amend_data_input(data):
    """If the data comes in without optional entries, add them.
    This doesn't translate to internal IDs for Dismod-AT. It rectifies
    the input, and this is how it should be saved or passed to another tool.
    """
    if data is None:
        return

    data = point_age_time_to_interval(data)

    if "name" not in data.columns:
        data = data.assign(name=data.index.astype(str))
    else:
        null_names = data[data.name.isnull()]
        if not null_names.empty:
            raise ValueError(f"There are some data values that lack data names. {null_names}")

    if "hold_out" not in data.columns:
        data = data.assign(hold_out=0)
    for additional in ["nu", "eta"]:
        if additional not in data.columns:
            data = data.assign(**{additional: nan})
    return data
