""" Functions for creating internal model representations of settings from EpiViv
"""

from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.rates import Smooth
from cascade.input_data.configuration import SettingsError
from cascade.core.context import ModelContext
from cascade.dismod.db.metadata import IntegrandEnum

RATE_TO_INTEGRAND = dict(
    iota=IntegrandEnum.Sincidence,
    rho=IntegrandEnum.remission,
    chi=IntegrandEnum.mtexcess,
    omega=IntegrandEnum.mtother,
    prevalence=IntegrandEnum.prevalence,
)


def initial_context_from_epiviz(configuration):
    context = ModelContext()
    context.parameters.modelable_entity_id = configuration.model.modelable_entity_id
    context.parameters.bundle_id = configuration.model.bundle_id
    context.parameters.gbd_round_id = configuration.gbd_round_id
    context.parameters.location_id = configuration.model.drill_location

    return context


def make_smooth(configuration, smooth_configuration):
    ages = smooth_configuration.age_grid
    if ages is None:
        ages = configuration.model.default_age_grid
    times = smooth_configuration.time_grid
    if times is None:
        times = configuration.model.default_time_grid
    grid = AgeTimeGrid(ages, times)

    d_time = PriorGrid(grid)
    d_age = PriorGrid(grid)
    value = PriorGrid(grid)

    d_age[:, :].prior = smooth_configuration.default.dage.prior_object
    d_time[:, :].prior = smooth_configuration.default.dtime.prior_object
    value[:, :].prior = smooth_configuration.default.value.prior_object

    if smooth_configuration.detail:
        for row in smooth_configuration.detail:
            if row.prior_type == "dage":
                pgrid = d_age
            elif row.prior_type == "dtime":
                pgrid = d_time
            elif row.prior_type == "value":
                pgrid = value
            else:
                raise SettingsError(f"Unknown prior type {row.prior_type}")
            pgrid[slice(row.age_lower, row.age_upper), slice(row.time_lower, row.time_upper)].prior = row.prior_object
    return Smooth(value, d_age, d_time)


def fixed_effects_from_epiviz(model_context, configuration):
    if configuration.rate:
        for rate_config in configuration.rate:
            rate_name = rate_config.rate
            if rate_name not in [r.name for r in model_context.rates]:
                raise SettingsError(f"Unspported rate {rate_name}")
            rate = getattr(model_context.rates, rate_name)
            rate.parent_smooth = make_smooth(configuration, rate_config)


def random_effects_from_epiviz(model_context, configuration):
    if configuration.random_effect:
        for smoothing_config in configuration.random_effect:
            rate_name = smoothing_config.rate
            if rate_name not in [r.name for r in model_context.rates]:
                raise SettingsError(f"Unspported rate {rate_name}")
            rate = getattr(model_context.rates, rate_name)
            location = smoothing_config.location
            rate.child_smoothings.append((location, make_smooth(configuration, smoothing_config)))
