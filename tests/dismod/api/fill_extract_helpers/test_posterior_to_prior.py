import pandas as pd
import numpy as np

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid
from cascade_at.model.utilities.integrand_grids import integrand_grids
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.model.utilities.grid_helpers import estimate_grid_from_draws


def test_get_prior_avgint_grid():

    settings = load_settings(BASE_CASE)
    alchemy = Alchemy(settings)

    rates = ['pini', 'iota', 'chi']

    grids = integrand_grids(alchemy=alchemy, integrands=rates)

    df = get_prior_avgint_grid(
        grids=grids,
        sexes=[1, 2],
        locations=[1]
    )
    assert type(df) == pd.DataFrame
    assert sorted(df['integrand_id'].unique()) == [0, 2, 7]
    assert all(df.location_id == 1)


def test_estimate_prior_grid():
    np.random.seed(0)
    settings = load_settings(BASE_CASE)
    alchemy = Alchemy(settings)

    prior = alchemy.get_smoothing_grid(rate=settings.rate[0])
    draws = np.abs(np.random.normal(loc=1e-3, scale=1e-7,
                                    size=(len(prior.ages), len(prior.times), 100)))
    estimate_grid_from_draws(
        ages=prior.ages, times=prior.times, draws=draws, grid_priors=prior.value
    )
    for (a, age), (t, time) in zip(enumerate(prior.ages), enumerate(prior.times)):
        assert prior.value[age, time].mean == draws[a, t, :].mean()


def test_estimate_priors_new_distribution():
    np.random.seed(0)
    settings = load_settings(BASE_CASE)
    alchemy = Alchemy(settings)

    prior = alchemy.get_smoothing_grid(rate=settings.rate[0])
    draws = np.abs(np.random.normal(loc=1e-3, scale=1e-7,
                                    size=(len(prior.ages), len(prior.times), 100)))
    estimate_grid_from_draws(
        ages=prior.ages, times=prior.times, draws=draws, grid_priors=prior.value,
        new_prior_distribution='uniform'
    )
    for (a, age), (t, time) in zip(enumerate(prior.ages), enumerate(prior.times)):
        assert prior.value[age, time].mean == draws[a, t, :].mean()
        assert prior.value[age, time].density == 'uniform'


def test_override_priors():
    np.random.seed(0)
    settings = load_settings(BASE_CASE)
    alchemy = Alchemy(settings)

    prior = alchemy.get_smoothing_grid(rate=settings.rate[0])
    draws = np.abs(np.random.normal(loc=1e-3, scale=1e-7,
                                    size=(len(prior.ages), len(prior.times), 100)))

    alchemy.override_priors(
        rate_grid=prior,
        update_dict={
            'value': draws,
            'dage': draws,
            'dtime': draws,
            'ages': prior.ages,
            'times': prior.times
        },
        new_prior_distribution=None
    )
    for (a, age), (t, time) in zip(enumerate(prior.ages), enumerate(prior.times)):
        assert prior.value[age, time].mean == draws[a, t, :].mean()
    for (a, age), (t, time) in zip(enumerate(prior.ages[:-1]), enumerate(prior.times)):
        assert prior.dage[age, time].mean == draws[a, t, :].mean()
    for (a, age), (t, time) in zip(enumerate(prior.ages), enumerate(prior.times[:-1])):
        assert prior.dtime[age, time].mean == draws[a, t, :].mean()


def test_apply_min_cv_to_value():
    settings = load_settings(BASE_CASE)
    alchemy = Alchemy(settings)

    prior = alchemy.get_smoothing_grid(rate=settings.rate[0]).value
    # Apply a ridiculously large coefficient of variation
    alchemy.apply_min_cv_to_prior_grid(prior_grid=prior, min_cv=1e6)
    for (a, age), (t, time) in zip(enumerate(prior.ages), enumerate(prior.times)):
        assert prior[age, time].standard_deviation == prior[age, time].mean * 1e6
