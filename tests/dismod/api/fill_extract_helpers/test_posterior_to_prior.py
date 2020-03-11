import pytest
import pandas as pd

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid


def test_get_prior_avgint_grid():
    prior_avgint_grid = get_prior_avgint_grid(
        settings=load_settings(BASE_CASE),
        integrands=['pini', 'iota', 'chi'],
        sexes=[1, 2],
        locations=[1]
    )
    assert type(prior_avgint_grid) == pd.DataFrame
    import pdb; pdb.set_trace()
    assert sorted(prior_avgint_grid.integrand_id.unique()) == [0, 2, 7]
