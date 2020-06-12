import pytest
from pathlib import Path
import numpy as np

from cascade_at.dismod.api.run_dismod import run_dismod
from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.dismod.api.dismod_extractor import DismodExtractorError


def test_empty_database():
    with pytest.raises(DismodExtractorError):
        DismodExtractor('temp-2.db')


def test_run_dismod_fit_predict(dismod, ihme, df):
    run = run_dismod(dm_file='temp.db', command='init')
    if run.exit_status:
        print(run.stderr)
    assert run.exit_status == 0
    run = run_dismod(dm_file='temp.db', command='fit fixed')
    if run.exit_status:
        print(run.stderr)
    assert run.exit_status == 0
    run = run_dismod(dm_file='temp.db', command='predict fit_var')
    if run.exit_status:
        print(run.stderr)
    assert run.exit_status == 0


def test_get_predictions(ihme, dismod, df):
    d = DismodExtractor(path=Path('temp.db'))
    pred = d.get_predictions()
    assert len(pred) == 484
    assert all(pred.columns == [
        'predict_id', 'sample_index', 'avgint_id', 'avg_integrand',
        'integrand_id', 'node_id', 'weight_id', 'subgroup_id', 'age_lower',
        'age_upper', 'time_lower', 'time_upper', 'c_age_group_id',
        'c_location_id', 'c_sex_id', 'c_year_id', 'x_0', 'x_1', 'x_2',
        'integrand_name', 'minimum_meas_cv', 'rate'
    ])


def test_format_for_ihme(ihme, dismod, df):
    d = DismodExtractor(path=Path('temp.db'))
    pred = d.format_predictions_for_ihme()
    # This prediction data frame is one longer for every location than the prediction
    # dataframe in `test_get_predictions` because there is an extra measure.
    assert len(pred) == 528
    assert all(pred.columns == [
        'location_id', 'age_group_id', 'year_id', 'sex_id', 'measure_id',
        'mean', 'upper', 'lower'
    ])

