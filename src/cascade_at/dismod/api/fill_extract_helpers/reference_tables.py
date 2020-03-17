import numpy as np
import pandas as pd
from numbers import Real

from cascade_at.dismod.constants import DensityEnum, IntegrandEnum, \
    RateEnum, enum_to_dataframe
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def default_integrand_table():
    df = pd.DataFrame({
        "integrand_name": enum_to_dataframe(IntegrandEnum)["name"],
        "minimum_meas_cv": 0.0
    })
    df = df.loc[df.integrand_name != 'incidence'].copy()
    return df


def default_rate_table():
    return pd.DataFrame({
        'rate_id': [rate.value for rate in RateEnum],
        'rate_name': [rate.name for rate in RateEnum],
        'parent_smooth_id': np.nan,
        'child_smooth_id': np.nan,
        'child_nslist_id': np.nan
    })


def construct_age_time_table(variable_name, variable, data_min=None, data_max=None):
    """
    Constructs the age or time table with age_id and age or time_id and time.
    Has unique identifiers for each.

    Parameters:
        variable_name: (str) one of 'age' or 'time'
        variable: (np.array) array of ages or times
        data_min: (float) minimum observed in the data
        data_max: (float) max observed in the data
    """
    LOG.info(f"Constructing {variable_name} table.")
    if data_min < np.min(variable):
        variable = np.append(variable, data_min)
    if data_max > np.max(variable):
        variable = np.append(variable, data_max)
    variable = variable[np.unique(variable.round(decimals=14), return_index=True)[1]]

    variable.sort()
    if variable[-1] - variable[0] < 1:
        variable = np.append(variable, variable[-1] + 1)
    df = pd.DataFrame(dict(id=range(len(variable)), var=variable))
    df.rename(columns={'id': f'{variable_name}_id', 'var': variable_name}, inplace=True)
    return df


def construct_node_table(location_dag):
    """
    Constructs the node table from a location
    DAG's to_dataframe() method.

    Parameters:
        location_dag: (cascade_at.inputs.locations.LocationDAG)
    """
    LOG.info("Constructing node table.")
    node = location_dag.to_dataframe()
    node = node.reset_index(drop=True)
    node["node_id"] = node.index
    p_node = node[["node_id", "location_id"]].rename(
        columns={"location_id": "parent_id", "node_id": "parent"}
    )
    node = node.merge(p_node, on="parent_id", how="left")
    node.rename(columns={
        "name": "node_name",
        "location_id": "c_location_id"
    }, inplace=True)
    node = node[['node_id', 'node_name', 'parent', 'c_location_id']]
    return node


def construct_covariate_table(covariates):
    """
    Constructs the covariate table from a list of Covariate objects.

    :param covariates: List(cascade_at.model.covariate.Covariate)
    :return: pd.DataFrame()
    """
    covariates_reordered = list()
    lookup = {search.name: search for search in covariates}
    for special in ["sex", "one"]:
        if special in lookup:
            covariates_reordered.append(lookup[special])
            del lookup[special]
    for remaining in sorted(lookup.keys()):
        covariates_reordered.append(lookup[remaining])
    LOG.info(f"Writing covariates {', '.join(c.name for c in covariates_reordered)}")

    null_references = list()
    for check_ref_col in covariates_reordered:
        if not isinstance(check_ref_col.reference, Real):
            null_references.append(check_ref_col.name)
    if null_references:
        raise ValueError(f"Covariate columns without reference values {null_references}.")

    covariate_rename = dict()
    for covariate_idx, covariate_obj in enumerate(covariates_reordered):
        covariate_rename[covariate_obj.name] = f"x_{covariate_idx}"

    covariate_table = pd.DataFrame({
        "covariate_id": np.arange(len(covariates_reordered)),
        "covariate_name": [covariate_rename[col.name] for col in covariates_reordered],
        "c_covariate_name": [col.name for col in covariates_reordered],
        "reference": np.array([col.reference for col in covariates_reordered], dtype=np.float),
        "max_difference": np.array([col.max_difference for col in covariates_reordered], dtype=np.float)
    })
    return covariate_table


def construct_density_table():
    return pd.DataFrame({
        'density_name': [x.name for x in DensityEnum]
    })
