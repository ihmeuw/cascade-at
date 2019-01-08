"""
Statistical operations on the model:

 * Creation of priors from posteriors.

"""
import numpy as np
from scipy.interpolate import RectBivariateSpline

from cascade.core.log import getLoggers
from cascade.dismod.db.metadata import RateName, IntegrandEnum

CODELOG, MATHLOG = getLoggers(__name__)


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
        posterior_draws (DataFrame): This dataframe has multiple copies of
            the ``fit_var`` table, each with a different fit for all model
            variables. The draws also contain merged columns from the ``var``
            table so that we know what the variables are for.

    Returns:
        None: It changes priors on the model context.
    """
    if posterior_draws is None or posterior_draws.empty:
        return
    _assign_rate_priors(model_context, posterior_draws)
    _assign_mulcov_priors(model_context, posterior_draws)


def _assign_rate_priors(model_context, posterior_draws):
    grandparent_id = model_context.parameters.grandparent_location_id
    parent_id = model_context.parameters.location_id
    underlying_rate = dict()
    random_effect = dict()
    rate_draws = posterior_draws[posterior_draws.covariate_id.isna()]
    # Posterior draws are copies of the var table, so they have sections
    # for each random field. The (smooth_id, node_id) uniquely identifies
    # each random field in the vars table.
    for (smooth_id, location_id), field_df in rate_draws.groupby(["smooth_id", "location_id"]):
        traits = field_df.iloc[0]
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


def _assign_mulcov_priors(model_context, posterior_draws):
    local_covariates = model_context.input_data.covariates
    mulcov_draws = posterior_draws[posterior_draws.covariate_id.notna()]
    for (smooth_id, location_id), field_df in mulcov_draws.groupby(["smooth_id", "location_id"]):
        # One of the covariate multipliers.
        traits = field_df.iloc[0]
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


def _assign_smooth_priors_from_random_effect(model_context, rate_name, underlying_df, random_effect_df):
    underlying_at = _estimates_from_one_grid(underlying_df)
    random_effect_at = _estimates_from_one_grid(random_effect_df)
    re = _dataframe_to_bivariate_spline(random_effect_at)
    adjusted_by_effect = underlying_at.apply(lambda row: row.mean * np.exp(re(row.age, row.time)))
    rate = underlying_at.assign(mean=adjusted_by_effect)
    _assign_smooth_priors_from_estimates(model_context.rates[rate_name].parent_smooth, rate)


def _dataframe_to_bivariate_spline(age_time_df):
    """Constructs a function which mimics how Dismod-AT turns a field of
    points in age and time into a continuous function.

    Args:
        age_time_df: Dataframe has columns age, time, and mean.

    Returns:
        function: Of age and time.
    """
    ordered = age_time_df.sort_values(["age", "time"])
    spline = RectBivariateSpline(*[ordered[n].values for n in ["age", "time", "mean"]], kx=1, ky=1)

    def bivariate_function(x, y):
        return spline(x, y)[0]

    return bivariate_function


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
