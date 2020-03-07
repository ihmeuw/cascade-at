import pytest
import pandas as pd

from cascade_at.settings.settings import load_settings
from cascade_at.settings.base_case import BASE_CASE
from cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior import get_prior_avgint_grid


def test_get_prior_avgint_dict():
    prior_avgint_dict = get_prior_avgint_grid(
        settings=load_settings(BASE_CASE),
        integrands=['prevalence', 'iota'],
        sexes=[1, 2],
        locations=[1]
    )
    for k, v in prior_avgint_dict.items():
        assert type(v) == pd.DataFrame
    assert ['prevalence', 'iota'] == list(prior_avgint_dict.keys())
