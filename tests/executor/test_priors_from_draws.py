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
import cascade.executor.priors_from_draws
from cascade.executor.priors_from_draws import (
    set_priors_from_parent_draws, estimate_grid_parameters, set_priors_from_draws
)
from cascade.model.priors import Uniform, Gaussian
from cascade.model.smooth_grid import SmoothGrid


def jitter(vars, rng):
    for group_name, group in vars.items():
        for key, var in group.items():
            jitter_one_grid(var, rng)


def jitter_one_grid(var, rng):
    for age, time in var.age_time():
        var[age, time] = (1 + 0.1 * (rng.uniform() - 0.5)) * var[age, time]


def test_priors_from_draws_fair(monkeypatch):
    """Stochastic draw construction."""
    # The goal is to check that the logic of which grids are applied is correct.
    seen = list()

    def _gather(grid_draws, ages, times):
        seen.append((grid_draws._group, grid_draws._key, grid_draws.__class__.__name__))
        a = np.full((len(ages), len(times), len(grid_draws)), 0.01, dtype=np.float)
        return a, a[:-1, :, :], a[:, :-1, :]

    monkeypatch.setattr(cascade.executor.priors_from_draws, "gather_draws_for_grid", _gather)

    rng = RandomState(2340238)
    draw_cnt = 3  # Can be long-running. Increase for focused testing.
    for i in range(5):
        args = parse_arguments(["z.db"])
        locations = nx.DiGraph()
        children = [4, 31, 64, 103, 137, 158, 166]
        locations.add_edges_from([(1, c) for c in children])
        settings = create_settings(rng, children)
        c = CascadePlan.from_epiviz_configuration(locations, settings, args)
        j = list(c.cascade_jobs)
        draws = None
        parent_model_has_random_effects = False

        for job in j:
            job_kind, job_args = c.cascade_job(job)
            if job_kind != "estimate_location":
                continue

            data = SimpleNamespace()
            data.locations = locations
            model = construct_model(data, job_args)

            # We aren't asking whether the values are correct but whether
            # the logic paths work.
            set_priors_from_parent_draws(model, draws)

            if draws is not None:
                base_rate_set = False
                for group, key, klass in seen:
                    if group == "rate" and klass == "RandomEffectDrawFunction":
                        base_rate_set = True
                    assert group != "random_effect", "random effects should never be set"
                if parent_model_has_random_effects:
                    assert base_rate_set, f"base rate unset {seen}"

            draws = list()
            for draw_idx in range(draw_cnt):
                var = model.var_from_mean()
                jitter(var, rng)
                draws.append(var)
            parent_model_has_random_effects = len(model.random_effect) > 0

            set_priors_from_draws(model, draws)

            seen.clear()


def test_estimate_grid_parameters_fair():
    """Tests for a single grid whether its priors are set in order."""
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