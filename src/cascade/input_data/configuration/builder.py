""" Functions for creating internal model representations of settings from EpiViz
"""
from types import MappingProxyType

import numpy as np
from scipy.special import logit

from cascade.input_data.configuration.construct_country import \
    covariate_records_from_settings, unique_country_covariate_transform
from cascade.input_data.configuration.id_map import PRIMARY_INTEGRANDS_TO_RATES, make_integrand_map
from cascade.model.covariates import Covariate, CovariateMultiplier
from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.priors import NO_PRIOR, Constant
from cascade.model.rates import Smooth
from cascade.input_data.configuration import SettingsError
from cascade.input_data.configuration.construct_study import (
    add_special_study_covariates,
    unique_study_covariate_transform,
    get_bundle_study_covariates,
)
from cascade.core.context import ModelContext
import cascade.model.priors as priors
from cascade.input_data import InputDataError

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def identity(x):
    return x


def squared(x):
    return np.power(x, 2)


def scale1000(x):
    return x * 1000


COVARIATE_TRANSFORMS = {0: identity, 1: np.log, 2: logit, 3: squared, 4: np.sqrt, 5: scale1000}
"""
These functions transform covariate data, as specified in EpiViz.
"""


class SettingsToModelError(InputDataError):
    """Error creating a model from the settings"""


def initial_context_from_epiviz(configuration):
    context = ModelContext()
    context.parameters.modelable_entity_id = configuration.model.modelable_entity_id
    context.parameters.bundle_id = configuration.model.bundle_id
    context.parameters.gbd_round_id = configuration.gbd_round_id
    context.parameters.location_id = configuration.model.drill_location
    context.parameters.rate_case = configuration.model.rate_case
    context.parameters.minimum_meas_cv = configuration.model.minimum_meas_cv
    context.parameters.global_data_eta = configuration.eta.data
    context.parameters.ode_step_size = configuration.model.ode_step_size

    context.policies = policies_from_settings(configuration)

    return context


def assign_covariates(model_context, covariate_record, transform_iterator, name_prefix=""):
    """
    The EpiViz interface allows assigning a covariate with a transformation
    to a specific target (rate, measure value, measure standard deviation).
    There will be one Dismod-AT covariate for each (input covariate dataset,
    transformation of that dataset).

    Args:
        model_context (ModelContext): model context that has age groups.
            The context is modified by this function. Covariate columns are
            added to input data and covariates are added to the list of
            covariates.
        covariate_record (CovariateRecord): The input covariate data as columns
        transform_iterator: An iterator that returns the covariate id and
            the numerical id of the function to use on it (log, squared).

    Returns:
        dictionary from the name and transformation to the covariate name.

    """
    covariate_map = {}  # to find the covariates for covariate multipliers.
    avgints = model_context.average_integrand_cases
    measurements = model_context.input_data.observations

    # This walks through all unique combinations of covariates and their
    # transformations. Then, later, we apply them to particular target
    # rates, meas_values, meas_stds.
    for covariate_id, transforms in transform_iterator:
        for transform in transforms:
            # This happens per application to integrand.
            settings_transform = COVARIATE_TRANSFORMS[transform]
            covariate_name = covariate_record.id_to_name[covariate_id]
            transform_name = settings_transform.__name__
            MATHLOG.info(f"Transforming {covariate_name} with {transform_name}")
            name = f"{name_prefix}{covariate_name}_{transform_name}"

            # The reference value is calculated from the download, not from the
            # the download as applied to the observations.
            reference = settings_transform(covariate_record.id_to_reference[covariate_id])
            covariate_obj = Covariate(name, reference, covariate_record.id_to_max_difference.get(covariate_id))
            model_context.input_data.covariates.append(covariate_obj)
            covariate_map[(covariate_id, transform)] = covariate_obj

            # Now attach the column to the observations.
            if measurements is not None:
                measurements[f"x_{name}"] = settings_transform(covariate_record.measurements[covariate_name])
            if avgints is not None:
                avgints[f"x_{name}"] = settings_transform(covariate_record.average_integrand_cases[covariate_name])

    return covariate_map


def settings_covariate_iter(config):
    """Iterate over both study and country covariate multipliers"""
    for mul_scov in config.study_covariate:
        yield mul_scov, mul_scov.study_covariate_id
    for mul_ccov in config.country_covariate:
        yield mul_ccov, mul_ccov.country_covariate_id


def create_covariate_multipliers(context, configuration, column_map):
    """
    Reads settings to create covariate multipliers. This attaches a
    covariate column with its reference value to a smooth grid
    and applies it to a rate value, integrand value, or integrand
    standard deviation. There aren't a lot of interesting choices in here.

    Args:
        context (ModelContext):
        configuration (Configuration): This describes the settings from EpiViz.
        column_map: Dictionary from covariate id and transform to the covariate
            object.
    """
    # Assumes covariates exist.
    gbd_to_dismod_integrand_enum = make_integrand_map()

    for mul_cov_config, cov_id in settings_covariate_iter(configuration):
        if mul_cov_config.measure_id not in gbd_to_dismod_integrand_enum:
            raise RuntimeError(f"The measure id isn't recognized as an integrand {mul_cov_config.measure_id}")
        target_dismod_name = gbd_to_dismod_integrand_enum[mul_cov_config.measure_id].name

        try:
            covariate_obj = column_map[(cov_id, mul_cov_config.transformation)]
        except KeyError:
            raise RuntimeError(f"A covariate id and its transformation weren't found: "
                               f"{cov_id}, with transform {mul_cov_config.transformation}.")

        if mul_cov_config.mulcov_type == "rate_value" and target_dismod_name in PRIMARY_INTEGRANDS_TO_RATES:
            target_name = PRIMARY_INTEGRANDS_TO_RATES[target_dismod_name]
        else:
            target_name = target_dismod_name
        smooth = make_smooth(configuration, mul_cov_config, name_prefix=f"{target_name}_{covariate_obj.name}")

        covariate_multiplier = CovariateMultiplier(covariate_obj, smooth)
        if mul_cov_config.mulcov_type == "rate_value":
            if target_dismod_name not in PRIMARY_INTEGRANDS_TO_RATES:
                raise SettingsToModelError(
                    f"Multiplier type for covariate {mul_cov_config.country_covariate_id} is on the rate value. "
                    f"Can only set a rate value on a primary integrand. Measure id "
                    f"{mul_cov_config.measure_id} name {target_dismod_name} is not a primary integrand. "
                    f"Primary integrands are {', '.join(list(sorted(PRIMARY_INTEGRANDS_TO_RATES.keys())))}"
                )
            target_rate = PRIMARY_INTEGRANDS_TO_RATES[target_dismod_name]
            MATHLOG.info(f"Covariate multiplier for measure_id {mul_cov_config.measure_id} "
                         f"applied to rate {target_rate} "
                         f"It was set to primary integrand {target_dismod_name} in EpiViz.")
            add_to_rate = getattr(context.rates, target_rate)
            add_to_rate.covariate_multipliers.append(covariate_multiplier)
        else:
            add_to_integrand = context.integrand_covariate_multipliers[target_dismod_name]
            if mul_cov_config.mulcov_type == "meas_value":
                add_to_integrand.value_covariate_multipliers.append(covariate_multiplier)
            elif mul_cov_config.mulcov_type == "meas_std":
                add_to_integrand.std_covariate_multipliers.append(covariate_multiplier)
            else:
                raise RuntimeError(f"mulcov_type isn't among the three {configuration.mulcov_type}")


def make_smooth(configuration, smooth_configuration, name_prefix=None):
    ages = smooth_configuration.age_grid
    if ages is None:
        if getattr(smooth_configuration, "rate", None) == "pini":
            ages = [0]
        else:
            ages = configuration.model.default_age_grid
    times = smooth_configuration.time_grid
    if times is None:
        times = configuration.model.default_time_grid
    grid = AgeTimeGrid(ages, times)

    # Age and time priors are optional so don't create a grid for them unless they are actually needed
    d_age = None
    d_time = None
    # But value will always have something, so set it up right away.
    value = PriorGrid(grid)

    smooth_name = name_prefix

    if name_prefix is not None:
        name_prefix += "_"
    else:
        name_prefix = ""

    if smooth_configuration.default.dage is not None:
        d_age = PriorGrid(grid)
        smooth_configuration.default.dage.prior_object.name = f"{name_prefix}dA"
        d_age[:, :].prior = smooth_configuration.default.dage.prior_object
    if smooth_configuration.default.dtime is not None:
        d_time = PriorGrid(grid)
        smooth_configuration.default.dtime.prior_object.name = f"{name_prefix}dT"
        d_time[:, :].prior = smooth_configuration.default.dtime.prior_object
    smooth_configuration.default.value.prior_object.name = f"{name_prefix}value"
    value[:, :].prior = smooth_configuration.default.value.prior_object

    if smooth_configuration.detail:
        for row in smooth_configuration.detail:
            if row.prior_type == "dage":
                if d_age is None:
                    d_age = PriorGrid(grid)
                    d_age[:, :].prior = priors.Uniform(float("-inf"), float("inf"), 0, name=f"{name_prefix}dA")
                pgrid = d_age
            elif row.prior_type == "dtime":
                if d_time is None:
                    d_time = PriorGrid(grid)
                    d_time[:, :].prior = priors.Uniform(float("-inf"), float("inf"), 0, name=f"{name_prefix}dT")
                pgrid = d_time
            elif row.prior_type == "value":
                pgrid = value
            else:
                raise SettingsError(f"Unknown prior type {row.prior_type}")
            row.prior_object.name = f"{name_prefix}{row.prior_type}" \
                                    f"_{row.age_lower}_{row.age_upper}_{row.time_lower}_{row.time_upper}"
            pgrid[slice(row.age_lower, row.age_upper), slice(row.time_lower, row.time_upper)].prior = row.prior_object
    return Smooth(value, d_age, d_time, name=smooth_name)


def build_constraint(constraint, name=None):
    """
    This makes a smoothing grid where the mean value is set to a given
    set of values. If records in the constraint have age-spans or time-spans,
    this reduces them to point values to place as priors on a grid.

    Args:
        constraint (pd.DataFrame): Measurements with age, time, mean values.

    Returns:
        Smooth: A smooth with priors on values, age differences, and time
        differences.
    """
    midpoint_constraint = constraint.assign(
        age=0.5 * (constraint["age_lower"] + constraint["age_upper"]),
        time=0.5 * (constraint["time_lower"] + constraint["time_upper"])
    )
    ages = midpoint_constraint["age"].tolist()
    times = midpoint_constraint["time"].tolist()
    grid = AgeTimeGrid(ages, times)
    smoothing_prior = PriorGrid(grid)
    smoothing_prior[:, :].prior = NO_PRIOR

    value_prior = PriorGrid(grid)
    # TODO: change the PriorGrid API to handle this elegantly
    for _, row in midpoint_constraint.iterrows():
        value_prior[row["age"], row["time"]].prior = Constant(row["mean"])
    return Smooth(value_prior, smoothing_prior, smoothing_prior, name=name)


def fixed_effects_from_epiviz(model_context, execution_context, configuration):
    if configuration.rate:
        for rate_config in configuration.rate:
            rate_name = rate_config.rate
            if rate_name not in [r.name for r in model_context.rates]:
                raise SettingsError(f"Unspported rate {rate_name}")
            rate = getattr(model_context.rates, rate_name)
            rate.parent_smooth = make_smooth(configuration, rate_config, name_prefix=rate_name)
    else:
        MATHLOG.info(f"No rates are configured.")

    # These are the _bundle_ study covariates. There are more observations
    # than those in the bundle because we add values.
    study_covariate_records = get_bundle_study_covariates(
        model_context, execution_context.parameters.bundle_id, execution_context,
        execution_context.parameters.tier)
    add_special_study_covariates(study_covariate_records, model_context, configuration.model.drill_sex)
    country_covariate_records = covariate_records_from_settings(
        model_context, execution_context, configuration, study_covariate_records)
    country_map = assign_covariates(
        model_context, country_covariate_records, unique_country_covariate_transform(configuration), "c_")
    study_map = assign_covariates(
        model_context, study_covariate_records, unique_study_covariate_transform(configuration), "s_")

    if set(country_map) & set(study_map):
        raise RuntimeError(f"The study covariate IDs and country covariate IDs collide "
                           f"for {set(country_map) & set(study_map)}.")
    all_id_map = {**country_map, **study_map}
    create_covariate_multipliers(model_context, configuration, all_id_map)


def random_effects_from_epiviz(model_context, configuration):
    if configuration.random_effect:
        for smoothing_config in configuration.random_effect:
            rate_name = smoothing_config.rate
            if rate_name not in [r.name for r in model_context.rates]:
                raise SettingsError(f"Unspported rate {rate_name}")
            rate = getattr(model_context.rates, rate_name)
            location = smoothing_config.location

            rate.child_smoothings.append(
                (location, make_smooth(configuration, smoothing_config, name_prefix=rate_name))
            )


def policies_from_settings(settings):
    return MappingProxyType(dict(settings.policies.items()))
