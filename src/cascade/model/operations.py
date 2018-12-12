"""
Statistical operations on the model:

 * Creation of priors from posteriors.

"""
from numpy import nan
import pandas as pd


def _var_grid_iterator(var_df):
    r"""Iterate over grids in the var table, returning the type of grid and the
    sub-section of the var table corresponding to that grid. There are

     * There is an MRF for every (``rate_id``, ``node_id``) combination.
       These are :math:`(\iota, \rho, \chi, \omega)` for each location.
     * There is an MRF for every (``covariate_id``, ``var_type``, ``integrand``)
       combination. These are the same covariate
       applied to the measured value, measured stdev, or the rate. These are the
       covariates :math:`(\alpha, \beta, \gamma)`.

    This function validates that we have an exhaustive list of what is in
    a ``smooth_grid``.

    Args:
        var_df (pd.DataFrame): The DB.var table from the Dismod-AT database.

    Returns:
        (str, (int, int), pd.DataFrame): the kind of grid, its unique
        identifier, and the grid itself.
    """
    for smooth_id, sub_grid_df in var_df.groupby("smooth_id"):
        rate_grids = sub_grid_df[
            (sub_grid_df.var_type == "rate") & (sub_grid_df.rate_id >= 0)][
            ["node_id", "rate_id"]].drop_duplicates()
        meas_value_full = sub_grid_df[(sub_grid_df.var_type == "mulcov_meas_value") & (sub_grid_df.covariate_id >= 0)]
        meas_value = meas_value_full[["integrand_id", "covariate_id"]].drop_duplicates()
        meas_std_full = sub_grid_df[(sub_grid_df.var_type == "mulcov_meas_std") & (sub_grid_df.covariate_id >= 0)]
        meas_std = meas_std_full[["integrand_id", "covariate_id"]].drop_duplicates()
        rate_value_full = sub_grid_df[(sub_grid_df.var_type == "mulcov_rate_value") & (sub_grid_df.covariate_id >= 0)]
        rate_value = rate_value_full[["rate_id", "covariate_id"]].drop_duplicates()

        categories = [rate_grids, meas_value, meas_std, rate_value]
        category_cnt = sum([not test_kind.empty for test_kind in categories])
        if category_cnt > 1:
            raise RuntimeError(
                f"The var table for smooth_id={smooth_id} should have only one kind "
                f"but seems to have multiple kinds: rate? {not rate_grids.empty} meas value {not meas_value.empty} "
                f"meas std? {not meas_std.empty} rate_value {not rate_value.empty}"
            )
        elif category_cnt == 0:
            raise RuntimeError(
                f"Cannot figure out why smooth_id={smooth_id} isn't one of "
                f"a rate or a covariate multiplier")
        # else: It's OK

        if not rate_grids.empty:
            node_id, rate_id = [int(rid) for rid in rate_grids.iloc[0]]
            yield "rate", (smooth_id, rate_id, node_id), sub_grid_df

        elif not meas_value.empty:
            integrand_id, covariate_id = [int(meas_id) for meas_id in meas_value.iloc[0]]
            yield "meas_value", (smooth_id, integrand_id, covariate_id), sub_grid_df

        elif not meas_std.empty:
            integrand_id, covariate_id = [int(std_id) for std_id in
                                          meas_std.iloc[0]]
            yield "meas_std", (smooth_id, integrand_id, covariate_id), sub_grid_df

        elif not rate_value.empty:
            rate_id, covariate_id = [int(rval_id) for rval_id in
                                     rate_value.iloc[0]]
            yield "rate_value", (smooth_id, rate_id, covariate_id), sub_grid_df

        else:
            raise RuntimeError(f"Grid has unknown type for smooth_id={smooth_id}")


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
     * the MRF for child effects

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
        pd.DataFrame: With parameters for each distribution.
    """
    # The strategy for breaking the work into parts is to split
    # the vars into different smooth_grids, process each separately,
    # and then recombine. Recombination requires renumbering priors,
    # so we will ensure that the tuple (smooth_id, prior_id) is unique.
    db = model_context.dismod_file
    strategies = dict()
    by_grid = list()
    for smooth_kind, smooth_params, grid_vars in _var_grid_iterator(db.var):
        strategy = strategies.get(smooth_kind, None)
        if strategy is None:
            continue

        smooth_id = smooth_params[0]
        prior, smooth, smooth_grid = estimate_single_grid(draws, db, smooth_id)
        by_grid.append([prior, smooth, smooth_grid])

    # Now concatenate and renumber all priors.
    return by_grid


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

    no_const_value = complete_grid.const_value.isna()
    not_squeezed = complete_grid.upper > complete_grid.lower
    not_constant = complete_grid[no_const_value & not_squeezed]

    # Iterate through the non-constant.
    value = draws.mean()
    std = draws.std()
    not_constant.loc[0]["mean"] = value
    not_constant.loc[0]["std"] = std

    # Change them back to being in the Dismod form.
    smooth, reduced_smooth_grid, reduced_prior = reduce_priors(smooth, complete_grid)
    return smooth, reduced_smooth_grid, reduced_prior
