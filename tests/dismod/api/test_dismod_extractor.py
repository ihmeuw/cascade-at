import pytest
import pandas as pd
import numpy as np

from cascade_at.dismod.api.run_dismod import run_dismod_commands
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.dismod.api.dismod_filler import DismodFiller
from cascade_at.dismod.api.dismod_extractor import DismodExtractor


@pytest.fixture(scope='module')
def df(mi, settings, temp_directory):
    alchemy = Alchemy(settings)
    d = DismodFiller(
        path=temp_directory / 'temp.db',
        settings_configuration=settings,
        measurement_inputs=mi,
        grid_alchemy=alchemy,
        parent_location_id=70,
        sex_id=2
    )
    d.fill_for_parent_child()
    return d


@pytest.fixture(scope='module')
def ex(temp_directory):
    d = DismodExtractor(path=temp_directory / 'temp.db')
    return d


def test_get_predictions(ex):
    import pdb; pdb.set_trace()
    run = run_dismod_commands(dm_file=str(ex.path), commands=['init', 'fit fixed', 'predict fit_var'])
    assert run.exit_status == 0
    pred = ex.get_predictions()


def test_format_for_ihme():
    pass