import pandas as pd

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid
from cascade_at.model.utilities.integrand_grids import integrand_grids
from cascade_at.model.grid_alchemy import Alchemy


def test_get_prior_avgint_grid():

    settings = load_settings(BASE_CASE)
    alchemy = Alchemy(settings)

    rates = ['pini', 'iota', 'chi']

    grids = integrand_grids(alchemy=alchemy, integrands=rates)

    prior_avgint_grid = get_prior_avgint_grid(
        grids=grids,
        sexes=[1, 2],
        locations=[1]
    )
    assert type(prior_avgint_grid) == pd.DataFrame
    assert sorted(prior_avgint_grid['integrand_id'].unique()) == [0, 2, 7]
