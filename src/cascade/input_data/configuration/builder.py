from cascade.model.grids import AgeTimeGrid, PriorGrid
import cascade.model.priors as priors


def apply_epiviz_configuration(model_context, configuration):
    smooth_grids_from_epiviz_configuration(model_context, configuration)
    return model_context


def _make_prior(density, lower=None, upper=None, mean=None, std=None, nu=None, eta=None):
    result = None
    try:
        if density == "uniform":
            result = priors.UniformPrior(float(lower), float(upper), float(mean))
        elif density == "gaussian":
            result = priors.Gaussian(float(mean), float(std), float(lower), float(upper))
    except (TypeError, ValueError):
        raise ValueError(
            f"Supplied pramaters not compatible with density '{density}':"
            f"{dict(lower=lower, upper=upper, mean=mean, std=std, nu=nu, eta=eta)}"
        )

    if result is None:
        raise ValueError(f"Unsuported density: '{density}'")

    return result


def smooth_grids_from_epiviz_configuration(model_context, random_effects_configuration):
    grid = AgeTimeGrid(
        random_effects_configuration["custom_age_grid"], random_effects_configuration["custom_time_grid"]
    )

    d_time = PriorGrid(grid)
    d_age = PriorGrid(grid)
    value = PriorGrid(grid)

    d_age[:, :] = _make_prior(**random_effects_configuration["default"]["dage"])
    d_time[:, :] = _make_prior(**random_effects_configuration["default"]["dtime"])
    value[:, :] = _make_prior(**random_effects_configuration["default"]["value"])

    for row in random_effects_configuration["detail"]:
        if row["prior_type"] == "age diff":
            pass
