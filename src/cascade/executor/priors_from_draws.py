import itertools as it
import numpy as np

from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def set_priors_from_draws(model, draws):
    """Sets priors from posteriors of the *same model*.
    Use this when you have fit fixed and then want to fit again for both.

    Args:
        model (Model): A complete model for this location. It will be modified.
        draws (List[Var]): A list of fits to this location.
    """
    if len(draws) == 0:
        return

    for group_name, group in model.items():
        if group_name not in draws[0]:
            continue

        for key, prior_grid in group.items():
            if key not in draws[0][group_name]:
                continue

            ages = prior_grid.ages
            times = prior_grid.times
            draw_value, draw_dage, draw_dtime = gather_draws_for_grid(
                draws, group_name, key, ages, times)

            estimate_grid_parameters(prior_grid.value, draw_value, ages, times)
            estimate_grid_parameters(prior_grid.dage, draw_dage, ages[:-1], times)
            estimate_grid_parameters(prior_grid.dtime, draw_dtime, ages, times[:-1])


def set_priors_from_parent_draws(model, draws):
    """Sets priors from posteriors of the *parent model*.

    Args:
        model (Model): A complete model for this location. It will be modified.
        draws (List[Var]): A list of fits to the parent of this location.
    """
    assert len(draws) > 0

    for group_name, group in model.items():
        if group_name not in draws[0] or group_name == "random_effect":
            continue

        for key, prior_grid in group.items():
            if key not in draws[0][group_name]:
                continue

            ages = prior_grid.ages
            times = prior_grid.times
            # This model's location_id is one of the children for the draws.
            if group_name == "rate" and (key, model.location_id) in draws[0]["random_effect"]:
                draw_value, draw_dage, draw_dtime = gather_draws_for_child_grid(
                    draws, group_name, key, ages, times, model.location_id)
                CODELOG.debug(f"Child prior found for {group_name} {key}")
            elif group_name != "rate":
                draw_value, draw_dage, draw_dtime = gather_draws_for_grid(draws, group_name, key, ages, times)
                CODELOG.debug(f"Prior found for {group_name} {key}")
            else:
                CODELOG.debug(f"No prior found for {group_name} {key}")
                continue

            estimate_grid_parameters(prior_grid.value, draw_value, ages, times)
            estimate_grid_parameters(prior_grid.dage, draw_dage, ages[:-1], times)
            estimate_grid_parameters(prior_grid.dtime, draw_dtime, ages, times[:-1])


def gather_draws_for_grid(draws, group_name, key, ages, times):
    # Gather data from incoming draws into an array of (draw, age, time)
    draw_data = np.zeros((len(draws), len(ages), len(times)))
    for didx in range(len(draws)):
        one_draw = draws[didx][group_name][key]
        for aidx, age in enumerate(ages):
            for tidx, time in enumerate(times):
                draw_data[didx, aidx, tidx] = one_draw(age, time)

    draw_data = draw_data.transpose([1, 2, 0])
    draw_dage = np.diff(draw_data, n=1, axis=0)
    draw_dtime = np.diff(draw_data, n=1, axis=1)
    return draw_data, draw_dage, draw_dtime


def gather_draws_for_child_grid(draws, group_name, key, ages, times, location_id):
    # Gather data from incoming draws into an array of (draw, age, time)
    # The form for the child with random effect comes from
    # https://bradbell.github.io/dismod_at/doc/avg_integrand.htm
    draw_data = np.zeros((len(draws), len(ages), len(times)))
    for didx in range(len(draws)):
        underlying = draws[didx][group_name][key]
        random_effect = draws[didx]["random_effect"][(key, location_id)]
        for aidx, age in enumerate(ages):
            for tidx, time in enumerate(times):
                draw_data[didx, aidx, tidx] = underlying(age, time) * np.exp(random_effect(age, time))

    draw_data = draw_data.transpose([1, 2, 0])
    draw_dage = np.diff(draw_data, n=1, axis=0)
    draw_dtime = np.diff(draw_data, n=1, axis=1)
    return draw_data, draw_dage, draw_dtime


def estimate_grid_parameters(grid_priors, draws, ages, times):
    for aidx, tidx in it.product(range(len(ages)), range(len(times))):
        age = ages[aidx]
        time = times[tidx]
        grid_priors[age, time] = grid_priors[age, time].mle(draws[aidx, tidx, :])
