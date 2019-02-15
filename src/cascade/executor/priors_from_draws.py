import itertools as it
import numpy as np

from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


def grids_to_set_priors(model, var):
    """Iterate over common parts of a DismodGroups object."""
    for group_name, group in model.items():
        if group_name not in var or group_name == "random_effect":
            continue

        for key, prior_grid in group.items():
            if key not in var[group_name]:
                continue

            yield group_name, key, prior_grid


def set_priors_from_draws(model, draws):
    """Sets priors from posteriors of the *same model*.
    Use this when you have fit fixed and then want to fit again for both.

    Args:
        model (Model): A complete model for this location. It will be modified.
        draws (List[DismodGroups]): A list of fits to this location.
    """
    if draws is None:
        return

    if len(draws) == 0:
        return

    for group_name, key, prior_grid in grids_to_set_priors(model, draws[0]):
        ages = prior_grid.ages
        times = prior_grid.times
        grid_draws = DrawFunction(draws, group_name, key)
        draw_value, draw_dage, draw_dtime = gather_draws_for_grid(grid_draws, ages, times)

        estimate_grid_parameters(prior_grid.value, draw_value, ages, times)
        estimate_grid_parameters(prior_grid.dage, draw_dage, ages[:-1], times)
        estimate_grid_parameters(prior_grid.dtime, draw_dtime, ages, times[:-1])


def set_priors_from_parent_draws(model, draws):
    """Sets priors from posteriors of the *parent model*.

    Args:
        model (Model): A complete model for this location. It will be modified.
        draws (List[DismodGroups]): A list of fits to the parent of this location.
    """
    if draws is None:
        return

    assert len(draws) > 0

    for group_name, key, prior_grid in grids_to_set_priors(model, draws[0]):
        ages = prior_grid.ages
        times = prior_grid.times
        # This model's location_id is one of the children for the draws.
        if group_name == "rate" and (key, model.location_id) in draws[0]["random_effect"]:
            grid_draws = RandomEffectDrawFunction(draws, group_name, key, model.location_id)
            draw_value, draw_dage, draw_dtime = gather_draws_for_grid(grid_draws, ages, times)
            CODELOG.debug(f"Child prior found for {group_name} {key}")
        elif group_name != "rate":
            grid_draws = DrawFunction(draws, group_name, key)
            draw_value, draw_dage, draw_dtime = gather_draws_for_grid(grid_draws, ages, times)
            CODELOG.debug(f"Prior found for {group_name} {key}")
        else:
            CODELOG.debug(f"No prior found for {group_name} {key}")
            continue

        estimate_grid_parameters(prior_grid.value, draw_value, ages, times)
        estimate_grid_parameters(prior_grid.dage, draw_dage, ages[:-1], times)
        estimate_grid_parameters(prior_grid.dtime, draw_dtime, ages, times[:-1])


class DrawFunction:
    """This says the child draw is the same as the source value."""
    def __init__(self, draws, group, key):
        self._draws = draws
        self._group = group
        self._key = key

    def __len__(self):
        return len(self._draws)

    def __call__(self, idx, age, time):
        return self._draws[idx][self._group][self._key](age, time)


class RandomEffectDrawFunction:
    """This applies rate = underlying x exp(random effect)."""
    def __init__(self, draws, group, key, location):
        self._draws = draws
        self._group = group
        self._key = key
        self._location = location

    def __len__(self):
        return len(self._draws)

    def __call__(self, idx, age, time):
        underlying = self._draws[idx][self._group][self._key](age, time)
        random_effect = self._draws[idx]["random_effect"][(self._key, self._location)](age, time)
        return underlying * np.exp(random_effect)


def gather_draws_for_grid(draws, ages, times):
    """Gather data from incoming draws into an array of (draw, age, time)

    Args:
        draws (DrawFunction): The draws are a list of Var fits.
        ages (np.ndarray): ages
        times (np.ndarray): times

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): 3 numpy arrays
        of shape (age, time, draws) where the second two have one-fewer ages
        and one-fewer times.
    """
    draw_data = np.zeros((len(draws), len(ages), len(times)))
    for didx in range(len(draws)):
        for aidx, age in enumerate(ages):
            for tidx, time in enumerate(times):
                draw_data[didx, aidx, tidx] = draws(didx, age, time)

    draw_data = draw_data.transpose([1, 2, 0])
    draw_dage = np.diff(draw_data, n=1, axis=0)
    draw_dtime = np.diff(draw_data, n=1, axis=1)
    return draw_data, draw_dage, draw_dtime


def estimate_grid_parameters(grid_priors, draws, ages, times):
    assert isinstance(draws, np.ndarray)
    assert len(draws.shape) == 3

    for aidx, tidx in it.product(range(len(ages)), range(len(times))):
        age = ages[aidx]
        time = times[tidx]
        grid_priors[age, time] = grid_priors[age, time].mle(draws[aidx, tidx, :])
