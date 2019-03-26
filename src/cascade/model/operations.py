"""
Statistical operations on the model:

 * Creation of priors from posteriors.

"""
from copy import copy

import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from cascade.core.log import getLoggers
from cascade.dismod.constants import IntegrandEnum, RateEnum
from cascade.model.priors import Constant

CODELOG, MATHLOG = getLoggers(__name__)


def set_priors_on_model_context(model_context, posterior_draws):
    """
    Given draws from the fit to a previous model, which will be a fit
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
    parent_id = model_context.parameters.parent_location_id
    underlying_rate = dict()
    random_effect = dict()
    rate_draws = posterior_draws[posterior_draws.covariate_id.isna()]
    # Posterior draws are copies of the var table, so they have sections
    # for each random field. The (smooth_id, node_id) uniquely identifies
    # each random field in the vars table.
    for unique_field, field_df in rate_draws.groupby(["smooth_id", "location_id"]):
        traits = field_df.iloc[0]
        rate_name = RateEnum(traits.rate_id).name
        # On the odd chance that the draws are for this location, passed into
        # itself again, the order of these if-then should check first for
        # the underlying rate because both grandparent and parent will match.
        if traits.location_id == grandparent_id:
            underlying_rate[rate_name] = field_df
        elif traits.location_id == parent_id:
            random_effect[rate_name] = field_df
        else:
            pass  # These are random effects that apply to siblings.
    for rate_name in underlying_rate.keys():
        _assign_smooth_priors_from_random_effect(
            model_context, rate_name, underlying_rate[rate_name], random_effect.get(rate_name, None))


def _assign_mulcov_priors(model_context, posterior_draws):
    local_covariates = model_context.input_data.covariates
    mulcov_draws = posterior_draws[posterior_draws.covariate_id.notna()]
    for unique_field, field_df in mulcov_draws.groupby(["smooth_id", "covariate_id"]):
        # One of the covariate multipliers.
        traits = field_df.iloc[0]
        if traits.var_type == "mulcov_rate_value":
            rate_name = RateEnum(traits.rate_id).name
            mulcovs = getattr(model_context.rates, rate_name).covariate_multipliers
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
    if random_effect_df is not None:
        random_effect_at = _estimates_from_one_grid(random_effect_df)
    else:
        random_effect_at = None
    _assign_smooth_priors_after_summary(model_context, rate_name, underlying_at, random_effect_at)


def _assign_smooth_priors_after_summary(model_context, rate_name, underlying_at, random_effect_at):
    if random_effect_at is not None:
        re = _dataframe_to_bivariate_spline(random_effect_at)
        adjusted_by_effect = underlying_at.apply(
            lambda row: row["mean"] * np.exp(re(row.age, row.time)), axis="columns")
        rate = underlying_at.assign(mean=adjusted_by_effect)
    else:
        rate = underlying_at
    rate_obj = getattr(model_context.rates, rate_name)
    _assign_smooth_priors_from_estimates(rate_obj.parent_smooth, rate)


def _dataframe_to_bivariate_spline(age_time_df):
    """Constructs a function which mimics how Dismod-AT turns a field of
    points in age and time into a continuous function.

    Args:
        age_time_df: Dataframe has columns age, time, and mean.

    Returns:
        function: Of age and time.
    """
    ordered = age_time_df.sort_values(["age", "time"])
    age = np.sort(np.unique(age_time_df.age.values))
    time = np.sort(np.unique(age_time_df.time.values))
    if len(age) == 1 and len(time) == 1:
        case = "constant"
        value = ordered["mean"].iloc[0]
    elif len(age) == 1:
        case = "time_only"
        spline = UnivariateSpline(time, ordered["mean"].values, k=1)
    elif len(time) == 1:
        case = "age_only"
        spline = UnivariateSpline(age, ordered["mean"].values, k=1)
    else:
        case = "both"
        spline = RectBivariateSpline(age, time, ordered["mean"].values.reshape(len(age), len(time)), kx=1, ky=1)

    def bivariate_function(x, y):
        if case == "constant":
            return value
        elif case == "time_only":
            return spline(y)
        elif case == "age_only":
            return spline(x)
        else:
            return spline(x, y)[0]

    return bivariate_function


MINIMUM_STANDARD_DEVIATION_ABSOLUTE = 1e-6
MINIMUM_STANDARD_DEVIATION_RELATIVE = 1e-3


def _assign_smooth_priors_from_estimates(smooth, estimate_at):
    """Sets value priors.
    If the model already set a constant, then leave it as it is.
    If the incoming posterior has a standard deviation below
    a threshold, then set the value as a constant.
    """
    value_priors = smooth.value_priors
    for row in estimate_at.itertuples():
        # If we don't copy the prior, then we modify in-place, which changes
        # the whole grid of priors by accident.
        prior = copy(value_priors[row.age, row.time].prior)
        assert prior is not None, f"none at ({row.age}, {row.time})"
        is_constant = "lower" in dir(prior) and prior.lower >= prior.upper
        is_constant |= isinstance(prior, Constant)
        if is_constant:
            continue

        std_ok = row.std > MINIMUM_STANDARD_DEVIATION_ABSOLUTE
        std_ok |= row.mean > 0 and row.std / row.mean > MINIMUM_STANDARD_DEVIATION_RELATIVE

        if std_ok:
            if hasattr(prior, "mean"):
                prior.mean = row.mean
            if hasattr(prior, "standard_deviation"):
                prior.standard_deviation = row.std
            value_priors[row.age, row.time].prior = prior
        else:
            value_priors[row.age, row.time].prior = Constant(prior.mean)


def _covariate_name_to_smooth(covariate_name, local_covariates, mulcovs):
    """Find in this model context the Smooth for a given covariate name."""
    covariate_objs = [cobj for cobj in local_covariates if cobj.name == covariate_name]
    if not covariate_objs:
        return None
    elif len(covariate_objs) > 1:
        raise RuntimeError(f"More than one covariate object for this covariate: {covariate_name}")
    match_mulcov = [mc for mc in mulcovs if mc.column == covariate_objs[0]]
    if not match_mulcov:
        return None
    elif len(match_mulcov) > 1:
        raise RuntimeError(f"More than one covariate multiplier matches this covariate: {covariate_objs[0]}")
    return match_mulcov[0].smooth


def _estimates_from_one_grid(field_df):
    """Given a dataframe with all var draws for a single field, return
    one var table with mean and standard deviation."""
    # Exclude mulstd to get just the grid values.
    exclude_mulstd = field_df[~field_df.var_type.str.startswith("mulstd")]
    grid_df = exclude_mulstd.set_index("fit_var_id")
    # Gives grid with one age-time for each var_id.
    with_at = grid_df[["age", "time"]].groupby(level=0).mean()
    var_only = grid_df[["fit_var_value"]].groupby(level=0)
    with_mean = var_only.mean().rename(columns={"fit_var_value": "mean"})
    with_std = var_only.std().rename(columns={"fit_var_value": "std"})
    # This makes columns: ["age", "time", "mean", "std"]
    return with_at.join(with_mean).join(with_std)
