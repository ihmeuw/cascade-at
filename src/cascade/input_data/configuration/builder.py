""" Functions for creating internal model representations of settings from EpiViv
"""
from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.rates import Smooth
from cascade.input_data.configuration import ConfigurationError
from cascade.core.context import ModelContext
from cascade.dismod.db.metadata import IntegrandEnum

RATE_TO_INTEGRAND = dict(
    iota=IntegrandEnum.Sincidence,
    rho=IntegrandEnum.remission,
    chi=IntegrandEnum.mtexcess,
    omega=IntegrandEnum.mtother,
    prevalence=IntegrandEnum.prevalence,
)

MEASURE_ID_TO_RATE_NAME = {
    6: "iota",  # Incidence
    7: "rho",  # Remission
    9: "chi",  # Excess Mortality
    16: "omega",
    18: "proportion",
    19: "continuous",
    38: "birth_prevalence",
}


def initial_context_from_epiviz(configuration):
    context = ModelContext()
    context.parameters.modelable_entity_id = configuration.model.modelable_entity_id
    context.parameters.bundle_id = configuration.model.bundle_id
    context.parameters.gbd_round_id = configuration.gbd_round_id
    context.parameters.location_id = configuration.model.drill_location

    return context


def fixed_effects_from_epiviz(model_context, configuration):
    for rate_config in configuration.rate:
        rate_name = MEASURE_ID_TO_RATE_NAME[rate_config.rate]
        if rate_name not in [r.name for r in model_context.rates]:
            raise ConfigurationError(f"Unspported rate {rate_name}")
        rate = getattr(model_context.rates, rate_name)
        ages = rate_config.age_grid
        if ages is None:
            ages = configuration.model.default_age_grid
        times = rate_config.time_grid
        if times is None:
            times = configuration.model.default_time_grid
        grid = AgeTimeGrid(ages, times)

        d_time = PriorGrid(grid)
        d_age = PriorGrid(grid)
        value = PriorGrid(grid)

        d_age[:, :].prior = rate_config.default.dage.prior_object
        d_time[:, :].prior = rate_config.default.dtime.prior_object
        value[:, :].prior = rate_config.default.value.prior_object

        if rate_config.detail:
            for row in rate_config.detail:
                if row.prior_type == "dage":
                    pgrid = d_age
                elif row.prior_type == "dtime":
                    pgrid = d_time
                elif row.prior_type == "value":
                    pgrid = value
                else:
                    raise ConfigurationError(f"Unknown prior type {row.prior_type}")
                pgrid[
                    slice(row.age_lower, row.age_upper), slice(row.time_lower, row.time_upper)
                ].prior = row.prior_object
        rate.parent_smooth = Smooth(value, d_age, d_time)


def integrand_grids_from_epiviz(model_context, configuration):
    ages = configuration.model.default_age_grid
    times = configuration.model.default_time_grid
    grid = AgeTimeGrid(ages, times)

    for rate in model_context.rates:
        if rate.parent_smooth:
            integrand = getattr(model_context.outputs.integrands, RATE_TO_INTEGRAND[rate.name].name)
            integrand.grid = grid
