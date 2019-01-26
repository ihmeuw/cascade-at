from math import nan

import pandas as pd

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

    # Some of the columns get the same treatment as the average integrand.
    # This takes care of integrand and location.
    like_avgint = write_avgint(dismod_file, data, covariate_rename).drop(columns=["avgint_id"])
    # Other columns have to do with priors.
    with_density = like_avgint.assign(density_id=like_avgint.density.apply(lambda x: DensityEnum[x].value))
    _check_column_assigned(with_density, "density")
    with_density = with_density.reset_index(drop=True).drop(columns=["density"])
    if "name" not in with_density.columns:
        with_density = with_density.assign(name=with_density.index.astype(str))
    elif not with_density.name.isnull().empty:
        raise RuntimeError(f"There are some data values that lack data names.")
    else:
        pass  # There are data names everywhere.
    if "hold_out" not in with_density.columns:
        with_density = with_density.assign(hold_out=0)
    for additional in ["nu", "eta"]:
        if additional not in with_density.columns:
            with_density = with_density.assign(**{additional: nan})
    for expand in ["age", "time"]:
        point_dimension = (f"{expand}_lower" not in with_density.columns and
                           f"{expand}_upper" not in with_density.columns)
        if point_dimension and expand in with_density.columns:
            with_density = with_density.assign(
                **{f"{expand}_lower": with_density[expand], f"{expand}_upper": with_density[expand]})
            with_density = with_density.drop(columns=[expand])

    dismod_file.data = with_density.rename(
        columns={"mean": "meas_value", "std": "meas_std", "name": "data_name"})


def write_avgint(dismod_file, avgint, covariate_rename):
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
    with_id = avgint.assign(integrand_id=avgint.integrand.apply(lambda x: IntegrandEnum[x].value))
    _check_column_assigned(with_id, "integrand")
    with_weight = with_id.assign(weight_id=with_id.integrand.apply(lambda x: INTEGRAND_TO_WEIGHT[x].value))
    with_weight = with_weight.drop(columns=["integrand"]).reset_index(drop=True)
    with_location = with_weight.merge(
        dismod_file.node[["c_location_id", "node_id"]], left_on="location", right_on="c_location_id") \
        .drop(columns=["c_location_id", "location"])
    with_location = with_location.rename(columns=covariate_rename)
    return with_location.assign(avgint_id=with_location.index)


def read_avgint(dismod_file):
    avgint = dismod_file.avgint
    with_integrand = avgint.assign(integrand=avgint.integrand_id.apply(lambda x: IntegrandEnum(x).name))
    with_location = with_integrand.merge(dismod_file.node, on="node_id", how="left") \
        .rename(columns={"c_location_id": "location"})
    return with_location[
        ["avgint_id", "location", "integrand", "age_lower", "age_upper", "time_lower", "time_upper"]]


def _check_column_assigned(with_id, column):
    column_id = f"{column}_id"
    if not with_id[with_id[column_id].isna()].empty:
        not_found_integrand = with_id[with_id[column_id].isna()][column].unique()
        kind_enum = globals()[f"{column.capitalize()}Enum"]
        err_message = (f"The {column} {not_found_integrand} weren't found in the "
                       f"{column} list {[i.name for i in kind_enum]}.")
        MATHLOG.error(err_message)
        raise RuntimeError(err_message)
