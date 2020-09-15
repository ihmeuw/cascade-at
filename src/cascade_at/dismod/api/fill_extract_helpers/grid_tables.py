import numpy as np
import pandas as pd
from typing import Dict

from cascade_at.core.log import get_loggers
from cascade_at.dismod.api.fill_extract_helpers import utils, reference_tables
from cascade_at.dismod.constants import DensityEnum, IntegrandEnum, \
    WeightEnum, MulCovEnum, RateEnum
from cascade_at.model.var import Var
from cascade_at.model.model import Model

LOG = get_loggers(__name__)

DEFAULT_DENSITY = ["uniform", 0, -np.inf, np.inf]


def construct_weight_grid_tables(weights: Dict[str, Var],
                                 age_df, time_df) -> (pd.DataFrame, pd.DataFrame):
    """
    Constructs the weight and weight_grid tables."

    Parameters
    ----------
    weights
        There are four kinds of weights:
        "constant", "susceptible", "with_condition", and "total".
        No other weights are used.
    age_df
        Age data frame from dismod db
    time_df
        Time data frame from dismod db

    Returns
    -------
    Tuple of the weight table and the weight grid table
    """
    LOG.info("Constructing weight and weight grid tables.")

    names = [w.name for w in WeightEnum]
    weight = pd.DataFrame({
        'weight_id': [w.value for w in WeightEnum],
        'weight_name': names,
        'n_age': [len(weights[name].ages) for name in names],
        'n_time': [len(weights[name].times) for name in names]
    })
    weight_grid = []
    for w in WeightEnum:
        LOG.info(f"Writing weight {w.name}.")
        one_grid = weights[w.name].grid[["age", "time", "mean"]].rename(columns={"mean": "weight"})
        one_grid["weight_id"] = w.value
        weight_grid.append(one_grid)
    weight_grid = pd.concat(weight_grid).reset_index(drop=True)

    weight_grid = utils.convert_age_time_to_id(
        df=weight_grid, age_df=age_df, time_df=time_df
    )
    weight_grid["weight_grid_id"] = weight_grid.index
    return weight, weight_grid


def _add_prior_smooth_entries(grid_name, grid, num_existing_priors, num_existing_grids,
                              age_df, time_df):
    """
    Adds prior smooth grid entries to the smooth grid table and any other tables
    it needs to be added to. Called from inside of ``construct_model_tables`` only.
    """
    age_count, time_count = (len(grid.ages), len(grid.times))
    prior_df = grid.priors
    assert len(prior_df) == (age_count * time_count + 1) * 3

    # Get the densities for the priors
    prior_df.loc[prior_df.density.isnull(), ["density", "mean", "lower", "upper"]] = DEFAULT_DENSITY
    prior_df["density_id"] = prior_df["density"].apply(lambda x: DensityEnum[x].value)
    prior_df["prior_id"] = prior_df.index + num_existing_priors
    prior_df["assigned"] = prior_df.density.notna()

    prior_df.rename(columns={"name": "prior_name"}, inplace=True)

    # Assign names to each of the priors
    null_names = prior_df.prior_name.isnull()
    prior_df.loc[~null_names, "prior_name"] = (
            prior_df.loc[~null_names, "prior_name"].astype(str) + "    " +
            prior_df.loc[~null_names, "prior_id"].astype(str)
    )
    prior_df.loc[null_names, "prior_name"] = prior_df.loc[null_names, "prior_id"].apply(
        lambda pid: f"{grid_name}_{pid}"
    )

    # Convert to age and time ID for prior table
    prior_df = utils.convert_age_time_to_id(
        df=prior_df, age_df=age_df, time_df=time_df
    )

    # Create the simple smooth data frame
    smooth_df = pd.DataFrame({
        "smooth_name": [grid_name],
        "n_age": [age_count],
        "n_time": [time_count],
        "mulstd_value_prior_id": [np.nan],
        "mulstd_dage_prior_id": [np.nan],
        "mulstd_dtime_prior_id": [np.nan]
    })

    # Create the grid entries
    # TODO: Pass in the value prior ID instead from posterior to prior
    long_table = prior_df.loc[prior_df.age_id.notna()][["age_id", "time_id", "prior_id", "kind"]]
    grid_df = long_table[["age_id", "time_id"]].sort_values(["age_id", "time_id"]).drop_duplicates()

    for kind in ["value", "dage", "dtime"]:
        grid_values = long_table.loc[long_table.kind == kind].drop("kind", axis="columns")
        grid_values.rename(columns={"prior_id": f"{kind}_prior_id"}, inplace=True)
        grid_df = grid_df.merge(grid_values, on=["age_id", "time_id"])

    grid_df = grid_df.sort_values(["age_id", "time_id"], axis=0).reindex()
    grid_df["const_value"] = np.nan
    grid_df["smooth_grid_id"] = grid_df.index + num_existing_grids

    prior_df = prior_df[[
        'prior_id', 'prior_name', 'lower', 'upper',
        'mean', 'std', 'eta', 'nu', 'density_id'
    ]].sort_values(by='prior_id').reset_index(drop=True)

    return prior_df, smooth_df, grid_df


def construct_subgroup_table() -> pd.DataFrame:
    """
    Constructs the default subgroup table. If we want to actually
    use the subgroup table, need to build this in.
    """
    return pd.DataFrame.from_dict({
        'subgroup_id': [0],
        'subgroup_name': ['world'],
        'group_id': [0],
        'group_name': ['world']
    })


def construct_model_tables(model: Model,
                           location_df: pd.DataFrame,
                           age_df: pd.DataFrame,
                           time_df: pd.DataFrame,
                           covariate_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Main function that loops through the items from a model object, which include
    rate, random_effect, alpha, beta, and gamma and constructs the modeling tables in dismod db.

    Each of these are "grid" vars, so they need entries in prior,
    smooth, and smooth_grid. This function returns those tables.

    It also constructs the rate, integrand, and mulcov tables (alpha, beta, gamma),
    plus nslist and nslist_pair tables.

    Parameters
    ----------
    model
        A model object that has rate information
    location_df
        A location / node data frame
    age_df
        An age data frame for dismod
    time_df
        A time data frame for dismod
    covariate_df
        A covariate data frame for dismod

    Returns
    -------
    A dictionary of data frames for each table name, includes:
        rate, prior, smooth, smooth_grid, mulcov, nslist, nslist_pair, and subgroup tables
    """
    nslist = {}
    smooth_table = pd.DataFrame()
    prior_table = pd.DataFrame()
    grid_table = pd.DataFrame()
    mulcov_table = pd.DataFrame()
    nslist_pair_table = pd.DataFrame()

    rate_table = reference_tables.default_rate_table()
    subgroup_table = construct_subgroup_table()

    covariate_index = dict(covariate_df[["c_covariate_name", "covariate_id"]].to_records(index=False))

    if "rate" in model:
        LOG.info("Adding rates...")
        for rate_name, grid in model["rate"].items():
            """
            Loop through each of the rates and add entries into the
            prior, and smooth tables. Also put an entry in the rate table so we know the
            parent smooth ID.
            """
            LOG.info(f"Adding rate {rate_name}")
            prior, smooth, grid = _add_prior_smooth_entries(
                grid_name=rate_name, grid=grid,
                num_existing_priors=len(prior_table),
                num_existing_grids=len(grid_table),
                age_df=age_df, time_df=time_df
            )

            smooth_id = len(smooth_table)
            smooth['smooth_id'] = smooth_id
            grid['smooth_id'] = smooth_id

            smooth_table = smooth_table.append(smooth)
            prior_table = prior_table.append(prior)
            grid_table = grid_table.append(grid)

            rate_table.loc[rate_table.rate_id == RateEnum[rate_name].value, "parent_smooth_id"] = smooth_id

    if "random_effect" in model:
        LOG.info("Adding random effects...")
        for (rate_name, child_location), grid in model["random_effect"].items():
            """
            Loop through each of the random effects and add entries
            into the prior and smooth tables.
            """
            LOG.info(f"Adding random effect for rate {rate_name}")
            grid_name = f"{rate_name}_re"
            if child_location is not None:
                grid_name = grid_name + f"_{child_location}"

            prior, smooth, grid = _add_prior_smooth_entries(
                grid_name=grid_name, grid=grid,
                num_existing_priors=len(prior_table),
                num_existing_grids=len(grid_table),
                age_df=age_df, time_df=time_df
            )

            smooth_id = len(smooth_table)
            smooth["smooth_id"] = smooth_id
            grid["smooth_id"] = smooth_id

            smooth_table = smooth_table.append(smooth)
            prior_table = prior_table.append(prior)
            grid_table = grid_table.append(grid)

            if child_location is None:
                rate_table.loc[rate_table.rate_id == RateEnum[rate_name].value, "child_smooth_id"] = smooth_id
            else:
                # If we are doing this for a child location, then we want to make entries in the
                # nslist and nslist_pair tables
                node_id = location_df[location_df.c_location_id == child_location].node_id.iloc[0]
                if rate_name not in nslist:
                    ns_id = len(nslist)
                    nslist[rate_name] = ns_id
                else:
                    ns_id = nslist[rate_name]
                rate_table.loc[rate_table.rate_id == RateEnum[rate_name].value, "child_nslist_id"] = ns_id
                nslist_pair_table = nslist_pair_table.append(pd.DataFrame({
                    'nslist_id': [ns_id],
                    'node_id': [node_id],
                    'smooth_id': [smooth_id]
                }))

    potential_mulcovs = ["alpha", "beta", "gamma"]
    mulcovs = [x for x in potential_mulcovs if x in model]

    for m in mulcovs:
        LOG.info(f"Looking for mulcovs {m}...")
        for (covariate, rate_or_integrand), grid in model[m].items():
            LOG.info(f"Adding covariate {covariate} on {rate_or_integrand}.")
            grid_name = f"{m}_{rate_or_integrand}_{covariate}"

            prior, smooth, grid = _add_prior_smooth_entries(
                grid_name=grid_name, grid=grid,
                num_existing_priors=len(prior_table),
                num_existing_grids=len(grid_table),
                age_df=age_df, time_df=time_df
            )
            smooth_id = len(smooth_table)
            smooth["smooth_id"] = smooth_id
            grid["smooth_id"] = smooth_id

            prior_table = prior_table.append(prior)
            smooth_table = smooth_table.append(smooth)
            grid_table = grid_table.append(grid)

            mulcov = pd.DataFrame({
                "mulcov_type": [MulCovEnum[m].value],
                "rate_id": [np.nan],
                "integrand_id": [np.nan],
                "covariate_id": [covariate_index[covariate]],
                "group_smooth_id": [smooth_id]
            })
            if m == "alpha":
                mulcov["rate_id"] = RateEnum[rate_or_integrand].value
            elif m in ["beta", "gamma"]:
                mulcov["integrand_id"] = IntegrandEnum[rate_or_integrand].value
            else:
                raise RuntimeError(f"Unknown mulcov type {m}.")
            mulcov_table = mulcov_table.append(mulcov)

    mulcov_table.reset_index(inplace=True, drop=True)
    mulcov_table["mulcov_id"] = mulcov_table.index
    mulcov_table["group_id"] = 0
    mulcov_table["subgroup_smooth_id"] = np.nan

    nslist_table = pd.DataFrame.from_records(
        data=list(nslist.items()),
        columns=["nslist_name", "nslist_id"]
    )
    nslist_pair_table.reset_index(inplace=True, drop=True)
    nslist_pair_table["nslist_pair_id"] = nslist_pair_table.index

    return {
        'rate': rate_table,
        'prior': prior_table,
        'smooth': smooth_table,
        'smooth_grid': grid_table,
        'mulcov': mulcov_table,
        'nslist': nslist_table,
        'nslist_pair': nslist_pair_table,
        'subgroup': subgroup_table
    }
