from itertools import product
from types import SimpleNamespace

import networkx as nx
import numpy as np
from numpy import inf
from numpy.random import RandomState

from cascade.executor.cascade_plan import CascadePlan
from cascade.executor.construct_model import construct_model
from cascade.executor.create_settings import create_settings
from cascade.executor.dismodel_main import parse_arguments
from cascade.executor.priors_from_draws import set_priors_from_parent_draws, estimate_grid_parameters
from cascade.model.priors import Uniform, Gaussian
from cascade.model.smooth_grid import SmoothGrid


def jitter(vars, rng):
    for group_name, group in vars.items():
        for key, var in group.items():
            jitter_one_grid(var, rng)


def jitter_one_grid(var, rng):
    for age, time in var.age_time():
        var[age, time] = (1 + 0.1 * (rng.uniform() - 0.5)) * var[age, time]


def test_priors_from_draws_fair():
    """Stochastic draw construction, can be long-running."""
    rng = RandomState(2340238)
    draw_cnt = 3
    for i in range(5):
        args = parse_arguments(["z.db"])
        locations = nx.DiGraph()
        locations.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (3, 6), (3, 7)])
        settings = create_settings(rng)
        c = CascadePlan.from_epiviz_configuration(locations, settings, args)
        j = list(c.cascade_jobs)
        draws = None
        for job in j:
            job_kind, job_args = c.cascade_job(job)
            data = SimpleNamespace()
            data.age_specific_death_rate = None
            model = construct_model(data, job_args)

            set_priors_from_parent_draws(model, draws)

            draws = list()
            for draw_idx in range(draw_cnt):
                var = model.var_from_mean()
                jitter(var, rng)
                draws.append(var)


def test_estimate_grid_parameters_fair():
    draw_cnt = 1000
    rng = RandomState(2340238)
    ages = np.linspace(0, 100, 5)
    times = np.linspace(2000, 2010, 3)
    priors = SmoothGrid(ages, times)
    priors.value[:, :] = Uniform(lower=1e-4, upper=1.5, mean=0.1)
    priors.dage[:, :] = Gaussian(lower=-5, upper=5, mean=0.1, standard_deviation=0.001)
    priors.dtime[:, :] = Gaussian(lower=-inf, upper=inf, mean=-0.1, standard_deviation=50)
    draws = np.zeros((len(ages), len(times), draw_cnt), dtype=np.float)
    for age_idx, time_idx in product(range(len(ages)), range(len(times))):
        draws[age_idx, time_idx, :] = rng.normal(
            loc=ages[age_idx] * 0.01 + (times[time_idx] - 2000) * 0.025,
            scale=0.01
        )
    estimate_grid_parameters(priors.value, draws, ages, times)
    estimate_grid_parameters(priors.dage, draws, ages[:-1], times)
    estimate_grid_parameters(priors.dtime, draws, ages, times[:-1])

    for grid in [priors.value, priors.dage, priors.dtime]:
        for age_idx, time_idx in product(range(len(ages) - 1), range(len(times) - 1)):
            age = ages[age_idx]
            time = times[time_idx]
            expected = age * 0.01 + (time - 2000) * 0.025
            found = grid[age, time]
            expected = min(found.upper, max(found.lower, expected))
            found_mean = found.mean
            assert np.isclose(expected, found_mean, atol=0.02, rtol=0.2), \
                f"at {age} {time} e {expected} f {found_mean}"
