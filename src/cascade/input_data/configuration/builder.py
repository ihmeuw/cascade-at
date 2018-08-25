from cascade.model.grids import AgeTimeGrid, PriorGrid
import cascade.model.priors as priors
from cascade.input_data.configuration import ConfigurationError


def apply_epiviz_configuration(model_context, configuration):
    smooth_grids_from_epiviz_configuration(model_context, configuration.random_effect)
    return model_context


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


def smooth_grids_from_epiviz_configuration(model_context, random_effects_configuration):
    for rate_config in random_effects_configuration:
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
