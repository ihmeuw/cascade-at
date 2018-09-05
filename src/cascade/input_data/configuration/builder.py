from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.rates import Smooth
import cascade.model.priors as priors
from cascade.input_data.configuration import ConfigurationError
from cascade.core.context import ModelContext
from cascade.input_data.db.configuration import from_epiviz
from cascade.input_data.db.bundle import bundle_with_study_covariates, freeze_bundle
from cascade.executor.no_covariate_main import bundle_to_observations
from cascade.dismod.db.metadata import IntegrandEnum
from cascade.input_data.configuration.form import Configuration

RATE_TO_INTEGRAND = dict(
    iota=IntegrandEnum.Sincidence,
    rho=IntegrandEnum.remission,
    chi=IntegrandEnum.mtexcess,
    omega=IntegrandEnum.mtother,
    prevalence=IntegrandEnum.prevalence,
)


def model_context_from_epiviz(execution_context):
    config_data = from_epiviz(execution_context)
    configuration = Configuration(config_data)
    errors = configuration.validate()
    if errors:
        import pdb

        pdb.set_trace()
    configuration.normalize()

    model_context = initial_context_from_epiviz(configuration)

    fixed_effects_from_epiviz(model_context, configuration.rate)

    freeze_bundle(execution_context)
    bundle, study_covariates = bundle_with_study_covariates(
        execution_context, bundle_id=model_context.parameters.bundle_id
    )
    model_context.inputs = bundle_to_observations(model_context.parameters, bundle)

    integrand_grids_from_epiviz(model_context, configuration)

    return model_context


def initial_context_from_epiviz(configuration):
    context = ModelContext()
    context.parameters.modelable_entity_id = configuration.model.modelable_entity_id
    context.parameters.bundle_id = configuration.model.bundle_id
    context.parameters.gbd_round_id = configuration.gbd_round_id
    context.parameters.location_id = configuration.model.drill_location

    return context


def _make_prior(config):
    result = None
    try:
        if config.density == "uniform":
            result = priors.UniformPrior(config.min, config.max, config.mean)
        elif config.density == "gaussian":
            result = priors.GaussianPrior(
                config.mean,
                config.std,
                config.min if config.min else float("-inf"),
                config.max if config.max else float("inf"),
            )
    except (TypeError, ValueError):
        raise ValueError(f"Supplied pramaters not compatible with density '{config.density}':" f"{config}")

    if result is None:
        raise ValueError(f"Unsuported density: '{config.density}'")

    return result


MEASURE_ID_TO_RATE_NAME = {
    6: "iota",  # Incidence
    7: "rho",  # Remission
    9: "chi",  # Excess Mortality
    16: "omega",
    18: "proportion",
    19: "continuous",
    38: "birth_prevalence",
}


def fixed_effects_from_epiviz(model_context, rates_configuration):
    for rate_config in rates_configuration:
        rate_name = MEASURE_ID_TO_RATE_NAME[rate_config.rate]
        if rate_name not in [r.name for r in model_context.rates]:
            raise ConfigurationError(f"Unspported rate {rate_name}")
        rate = getattr(model_context.rates, rate_name)
        ages = rate_config.age_grid
        times = rate_config.time_grid
        grid = AgeTimeGrid(ages, times)

        d_time = PriorGrid(grid)
        d_age = PriorGrid(grid)
        value = PriorGrid(grid)

        d_age[:, :].prior = _make_prior(rate_config.default.dage)
        d_time[:, :].prior = _make_prior(rate_config.default.dtime)
        value[:, :].prior = _make_prior(rate_config.default.value)

        for row in rate_config.detail:
            if row.prior_type == "dage":
                pgrid = d_age
            elif row.prior_type == "dtime":
                pgrid = d_time
            elif row.prior_type == "value":
                pgrid = value
            else:
                raise ConfigurationError(f"Unknown prior type {row.prior_type}")
            pgrid[slice(row.age_lower, row.age_upper), slice(row.time_lower, row.time_upper)].prior = _make_prior(row)
        rate.parent_smooth = Smooth(value, d_age, d_time)


def integrand_grids_from_epiviz(model_context, configuration):
    ages = configuration.model.default_age_grid
    times = configuration.model.default_time_grid
    grid = AgeTimeGrid(ages, times)

    for rate in model_context.rates:
        if rate.parent_smooth:
            integrand = getattr(model_context.outputs.integrands, RATE_TO_INTEGRAND[rate.name].name)
            integrand.grid = grid
