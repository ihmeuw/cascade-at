"""
Statistical operations on the model:

 * Creation of priors from posteriors.

"""
from math import isnan, nan
import pandas as pd

from cascade.core.log import getLoggers
from cascade.dismod.db.metadata import RateName, IntegrandEnum

CODELOG, MATHLOG = getLoggers(__name__)


def random_field_iterator(var_df):
    r"""Iterate over random fields in the var table, returning the type of grid and the
    sub-section of the var table corresponding to that grid.

     * There is an MRF for every (``rate_id``, ``node_id``) combination,
       where the node is the parent or its children, none of the other
       nodes in the node table.
       These are :math:`(\iota, \rho, \chi, \omega)` for each location.
     * There is an MRF for every (``covariate_id``, ``var_type``, ``integrand``)
       combination. These are the same covariate
       applied to the measured value, measured stdev, or the rate. These are the
       covariates :math:`(\alpha, \beta, \gamma)`.

    Args:
        var_df (pd.DataFrame): The DB.var table from the Dismod-AT database.

    Returns:
        (str, (int, int), pd.DataFrame): the kind of grid, its unique
        identifier, and the grid itself.
    """
    # groupby ignores every record where any of the keys are Null, so it
    # only gets the rates.
    for rate_index, sub_grid_df in var_df.groupby(["smooth_id", "rate_id", "node_id"]):
        smooth_id, rate_id, node_id = rate_index
        yield "rate", (smooth_id, rate_id, node_id), sub_grid_df

    for integrand_index, sub_grid_df in var_df.groupby(["smooth_id", "integrand_id", "covariate_id"]):
        smooth_id, integrand_id, covariate_id = integrand_index
        var_type = sub_grid_df.iloc[0].var_type
        if var_type == "mulcov_meas_value":
            kind = "meas_value"
        elif var_type == "mulcov_meas_std":
            kind = "meas_std"
        else:
            raise RuntimeError(f"Unknown var_type {var_type} in var table")
        yield kind, (smooth_id, integrand_id, covariate_id), sub_grid_df

    for alpha_index, sub_grid_df in var_df.groupby(["smooth_id", "rate_id", "covariate_id"]):
        smooth_id, rate_id, covariate_id = alpha_index
        yield "rate_value", (smooth_id, rate_id, covariate_id), sub_grid_df


def set_priors_on_model_context(model_context, posterior_draws):
    """
    Given draws from the fit to a previous model, which will the a fit
    to the parent location, set parameters on this model, which is a
    child location. The priors set will be:

     * Rates - These are determined by the rate and random effect of
       the parent.

     * Random Effects - These are not set because the parent cannot inform
       the random effect of its grandchildren.

     * Covariate Multipliers - These are the alpha, beta, and gamma,
       and they will be set from parent's draws.

    Any prior distributions that are fixed in the model are not set.
    If the posterior draw is fixed, then it will be fixed in this model.

    Args:
        model_context (ModelContext): The Model.
        posterior_draws (pd.DataFrame): This dataframe has multiple copies of
            the ``fit_var`` table, each with a different fit for all model
            variables. The draws also contain merged columns from the ``var``
            table so that we know what the variables are for.

    Returns:
        None: It changes priors on the model context.
    """
    # Posterior draws are copies of the var table, so they have sections
    # for each random field. The (smooth_id, node_id) uniquely identifies
    # each random field in the vars table.
    grandparent_id = model_context.parameters.grandparent_location_id
    parent_id = model_context.parameters.location_id
    local_covariates = model_context.input_data.covariates
    underlying_rate = dict()
    random_effect = dict()

    for (smooth_id, location_id), field_df in posterior_draws.groupby(["smooth_id", "location_id"]):
        # One of the covariate multipliers.
        traits = field_df.iloc[0]
        if not isnan(traits.covariate_id):
            if traits.var_type == "mulcov_rate_value":
                rate_name = RateName(traits.rate_id).name
                mulcovs = model_context.rates[rate_name].covariate_multipliers
            elif traits.var_type == "mulcov_meas_value":
                integrand_name = IntegrandEnum(traits.integrand_id).name
                mulcovs = model_context.integrand_covariate_multipliers[integrand_name].value_covariate_multipliers
            elif traits.var_type == "mulcov_meas_std":
                integrand_name = IntegrandEnum(traits.integrand_id).name
                mulcovs = model_context.integrand_covariate_multipliers[integrand_name].std_covariate_multipliers
            else:
                raise RuntimeError(f"Var type {traits.var_type} instead of a mulcov.")

            smooth = _covariate_name_to_smooth(traits.covariate_name, local_covariates, mulcovs)
            if smooth:
                estimate_at = _estimates_from_one_grid(field_df)
                _assign_smooth_priors_from_estimates(smooth, estimate_at)
        else:
            rate_name = RateName(traits.rate_id).name
            if traits.location_id == parent_id:
                random_effect[rate_name] = field_df
            elif traits.location_id == grandparent_id:
                underlying_rate[rate_name] = field_df
            else:
                pass  # These are random effects that apply to siblings.

    for rate_name in underlying_rate.keys():
        _assign_smooth_priors_from_random_effect(
            model_context, rate_name, underlying_rate[rate_name], random_effect.get(rate_name, None))


def _assign_smooth_priors_from_random_effect(model_context, rate_name,
                                             underlying_df, random_effect_df):
    pass


def _assign_smooth_priors_from_estimates(smooth, estimate_at):
    value_priors = smooth.value_priors
    for row in estimate_at.itertuples():
        prior = value_priors[row.age, row.time]
        prior.mean = row.mean
        prior.std = row.std
        value_priors[row.age, row.time] = prior


def _covariate_name_to_smooth(covariate_name, local_covariates, mulcovs):
    """Find in this model context the Smooth for a given covariate name."""
    covariate_objs = [cobj for cobj in local_covariates if cobj.name == covariate_name]
    if not covariate_objs:
        return None
    match_mulcov = [mc for mc in mulcovs if mc.column == covariate_objs[0]]
    if not match_mulcov:
        return None
    return match_mulcov[0].smooth


def _estimates_from_one_grid(field_df):
    """Given a dataframe with all var draws for a single field, return
    one var table with mean and standard deviation."""
    # Exclude mulstd to get just the grid values.
    exclude_mulstd = field_df[~field_df.var_type.str.startswith("mulstd")]
    grid_df = exclude_mulstd.set_index("fit_var_id")
    with_at = grid_df[["age", "time"]]
    var_only = grid_df[["fit_var_value"]].groupby(level=0)
    with_mean = var_only.mean().rename(columns={"fit_var_value": "mean"})
    with_std = var_only.std().rename(columns={"fit_var_value": "std"})
    # This makes columns: ["age", "time", "mean", "std"]
    return with_at.join(with_mean).join(with_std)


def estimate_priors_from_posterior_draws(draws, model_context, execution_context):
    r"""
    Given a dataframe of sampled outcomes from a previous run of DismodAT,
    find parameters for the given distributions. This works from the model
    context. The draws are a representation of :math:`p(\theta, u|y,\eta)`
    where :math:`\eta` are the parameters for the prior distributions on
    :math:`\theta`, the model variables, :math:`y` are the data,
    and :math:`u` are the random effects. The goal of this function is to
    calculate the likelihood of the parameters for the prior distribution
    of the next step down in the cascade,

    .. math::

        p(\eta|y) = \int p(\eta|\theta) p(\theta|y) d\theta

    and use that to generate one value of the most likely priors,
    :math:`\eta^*` that will initialize the next step down.

    The priors in this problem are on

     * the Markov Random Field (MRF) for each rate.
     * the MRF for the covariates, :math:`(\alpha,\beta,\gamma)`
     * the MRF for random effects

    Each MRF includes hyper-priors :math:`\lambda` on the standard deviations,
    and most of the priors are the value priors, age differences, and
    time differences.

    Args:
        draws (pd.DataFrame): Has ``fit_var_id``, ``fit_var_value``,
            ``residual_value``, ``residual_dage``, ``residual_dtime``,
            ``sample_index``. Where the residuals can be NaN. The
            zeroth sample is the initial fit, aka the MAP estimate.
            The other samples are samples around that fit.

        model_context: The Model.

        execution_context: Where to find the Dismod File.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): Three tables for
        the DismodFile: the smooth, smooth_grid, and prior tables.
        Parameters in the prior tables are set, and hyper-parameters
        in the smooth are set. The smooth ids and smooth_grid ids should
        be the same, but the priors will be renumbered and expanded.
    """
    # The strategy for breaking the work into parts is to split
    # the vars into different smooth_grids, process each separately,
    # and then recombine. Recombination requires renumbering priors,
    # so we will ensure that the tuple (smooth_id, prior_id) is unique.
    db = model_context.dismod_file
    smooths = list()
    grids = list()
    priors = list()
    for smooth_kind, smooth_params, grid_vars in random_field_iterator(db.var):
        smooth_id = smooth_params[0]
        smooth, smooth_grid, prior = estimate_single_grid(draws, db, smooth_id)
        smooths.append(smooth)
        grids.append(smooth_grid)
        priors.append(prior.assign(smooth_id=smooth_id))

    all_smooth, all_grids, final_priors = concatenate_grids_and_priors(smooths, grids, priors)

    return all_smooth, all_grids, final_priors


def concatenate_grids_and_priors(smooths, grids, priors):
    all_smooth = pd.concat(smooths, axis=0, sort=False)
    all_grids = pd.concat(grids, axis=0, sort=False)
    all_priors = pd.concat(priors, axis=0, sort=False)
    # The prior_ids are repeated, but (prior_id, smooth_id) is unique.
    indexed_priors = all_priors.assign(one_id=list(range(len(all_priors))))
    prior_index = indexed_priors[["prior_id", "smooth_id", "one_id"]]
    final_priors = indexed_priors.drop(["smooth_id", "prior_id"],
                                       axis=1).rename({"one_id": "prior_id"},
                                                      axis=1)
    for kind in ["value", "dage", "dtime"]:
        id_col = f"{kind}_prior_id"
        grids_indexed = all_grids.merge(
            prior_index, how="left", left_on=[id_col, "smooth_id"],
            right_on=["prior_id", "smooth_id"])
        all_grids = grids_indexed.drop([id_col, "prior_id"], axis=1).rename(
            {"one_id": id_col})
    return all_smooth, all_grids, final_priors


def expand_priors(smooth_df, grid_df, prior_df):
    """
    Given a subset of a smooth grid, return a dataframe where the value,
    dage, and dtime priors are broken out into a prior_type column
    with the value "value", "dage", or "dtime".

    Args:
        grid_df (pd.DataFrame): This is the `DB.smooth_grid` table but it has
            the priors on it. The ``density_id`` tells you whether it is nan.
            The ``prior_id`` is dropped.
    """
    prior_kinds = ["value", "dage", "dtime"]
    hstacked = list()
    for kind in prior_kinds:
        other_ids = [f"{pstring}_prior_id" for pstring in prior_kinds if pstring is not kind]
        generic_prior = grid_df.drop(other_ids, axis=1) \
            .rename({f"{kind}_prior_id": "prior_id"}, axis=1) \
            .assign(prior_type=kind)
        with_priors = generic_prior.merge(prior_df, how="left")
        hstacked.append(with_priors)

    # Hyper-priors are included with the mulstd_% kind.
    for std_kind in prior_kinds:
        renamed = smooth_df.rename({f"mulstd_{std_kind}_prior_id": "prior_id"}, axis=1)[["prior_id"]]
        if renamed.loc[0].isna().prior_id:
            std_addition = pd.DataFrame({"prior_type": [f"mulstd_{std_kind}_prior"], "prior_id": -1})
        else:
            std_addition = renamed.merge(prior_df, how="left") \
                .assign(prior_type=f"mulstd_{std_kind}_prior")
        hstacked.append(std_addition)

    return pd.concat(hstacked, axis=0, sort=False).drop("prior_id", axis=1)


def reduce_priors(smooth, grid_with_priors):
    """
    The priors are expanded into a dataframe that has a kind column
    that specifies value, or dage, or dtime. We want to take that and
    create priors.

    Args:
        smooth: This is the single row from the smooth table.
        grid_with_priors: smooth_grid and prior table joined, without IDs,
            and expanded so a separate row for the value prior, dage prior,
            and dtime prior.
    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): The smooth, smooth_grid,
        and priors tables, as they would go into the Dismod file.
    """
    # Add prior_id to all lines that have a density, but leave other
    # prior_id as nan. The result will be aligned between priors and smooth_grid.
    aligned = grid_with_priors.assign(prior_id=nan)
    has_prior = aligned.density_id.notna()
    aligned.loc[has_prior, "prior_id"] = list(range(has_prior.sum()))

    kinds = ["value", "dage", "dtime"]

    # Set the smooth priors, which may be nan.
    for std_kind in kinds:
        prior_type = f"mulstd_{std_kind}_prior"  # noqa: F841
        smooth.loc[:, f"mulstd_{std_kind}_prior_id"] = aligned.query("prior_type == @prior_type").prior_id

    # Separate the priors from the main dataframe because they need to be
    # returned as a separate table.
    priors_columns = [
        "prior_id", "prior_name", "lower", "upper", "mean", "std", "eta", "no", "density_id"
    ]
    priors_has = [pcol for pcol in priors_columns if pcol in aligned.columns]
    priors_including_nans = aligned[priors_has]
    priors = priors_including_nans[priors_including_nans.prior_id.notna()]

    # Pare down the smooth_grid to just its columns.
    smooth_grid_columns = [
        "smooth_grid_id", "const_value", "prior_type", "prior_id", "smooth_id", "age_id", "time_id"
    ]
    smooth_grid_has = list(set(smooth_grid_columns) & set(aligned.columns))
    expanded_grid = aligned[smooth_grid_has]
    reduced_grid = expanded_grid.drop(["prior_type", "prior_id"], axis=1).drop_duplicates()

    # Then reassemble it with separate columns for different prior_ids.
    for prior_kind in kinds:
        kind_column = f"{prior_kind}_prior_id"
        by_kind = expanded_grid[expanded_grid.prior_type == prior_kind].rename({"prior_id": kind_column}, axis=1)
        reduced_grid = reduced_grid.merge(by_kind[["smooth_grid_id", kind_column]], on="smooth_grid_id")

    return smooth, reduced_grid, priors


def next_age_and_time(smooth_grid, age, time):
    ordered_ages = pd.DataFrame({"age_id": smooth_grid["age_id"].unique()}) \
        .merge(age, how="left") \
        .sort_values(by="age")
    ordered_ages["next_age_id"] = ordered_ages["age_id"].shift(periods=-1).fillna(-1).astype(int)
    ordered_times = pd.DataFrame({"time_id": smooth_grid["time_id"].unique()}) \
        .merge(time, how="left") \
        .sort_values(by="time")
    ordered_times["next_time_id"] = ordered_times["time_id"].shift(periods=-1).fillna(-1).astype(int)
    return ordered_ages, ordered_times


def estimate_single_grid(draws, dismod_file, smooth_id):
    """Finds draws for the given smooth id. Constructs new prior parameters
    for that smooth id. Returns two dataframes to describe the priors.
    Call this function for each smooth id. Then make the ``prior_id``
    values non-overlapping and concatenate them to get the priors list.

    Args:
        draws (pd.DataFrame): Draws with ``fit_var_id``, ``fit_var_value``,
            and ``sample_index``.
        dismod_file: The Dismod db file.
        smooth_id (int): The smoothing for which to do this work.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame): First dataframe is the prior table
        for just this smooth_id. It is one-indexed. Second dataframe
        is the row of the smooth table with priors filled out. The third dataframe
        is the smooth_grid for this smooth_id. The ``smooth_grid_id``
        values are correct, but the priors point to the one-based priors
        that are returned.
    """
    smooth = dismod_file.smooth[dismod_file.smooth.smooth_id == smooth_id].copy()
    smooth_grid = dismod_file.smooth_grid[dismod_file.smooth_grid.smooth_id == smooth_id]
    next_age, next_time = next_age_and_time(smooth_grid, dismod_file.age, dismod_file.time)
    smooth_grid_linked = smooth_grid.merge(next_age, how="left").merge(next_time, how="left")
    # Expand priors into one column. Merge with the priors.
    complete_grid = expand_priors(smooth, smooth_grid_linked, dismod_file.prior)
    # With priors expanded, we can write to them. This is where we estimate.
    no_const_value = complete_grid.const_value.isna()
    not_squeezed = complete_grid.upper > complete_grid.lower
    not_constant = complete_grid[no_const_value & not_squeezed]

    # Iterate through the non-constant, and look at value priors.
    mutable_value = not_constant[not_constant.prior_type == "value"]
    var = dismod_file.var
    for index, row in mutable_value.iterrows():
        age_id = row["age_id"]
        time_id = row["time_id"]
        var_id = var[(var.smooth_id == smooth_id) & (var.age_id == age_id) & (var.time_id == time_id)].var_id.values[0]
        sub_draws = draws[draws.fit_var_id == var_id]
        value = sub_draws.fit_var_value.mean()
        std = sub_draws.fit_var_value.std()
        complete_grid.loc[(complete_grid.age_id == age_id) & (complete_grid.time_id == time_id), "mean"] = value
        complete_grid.loc[(complete_grid.age_id == age_id) & (complete_grid.time_id == time_id), "std"] = std

    # Change them back to being in the Dismod form.
    smooth, reduced_smooth_grid, reduced_prior = reduce_priors(smooth, complete_grid)
    return smooth, reduced_smooth_grid, reduced_prior
