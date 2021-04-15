from pathlib import Path

import pytest

from cascade_at.dismod.api.dismod_extractor import DismodExtractor
from cascade_at.dismod.api.dismod_extractor import DismodExtractorError
from cascade_at.dismod.api.run_dismod import run_dismod


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
    pred = d._extract_raw_predictions()
    assert len(pred) == 484
    assert all(pred.columns == [
        'predict_id', 'sample_index', 'avgint_id', 'avg_integrand',
        'integrand_id', 'node_id', 'weight_id', 'subgroup_id', 'age_lower',
        'age_upper', 'time_lower', 'time_upper', 'c_age_group_id',
        'c_location_id', 'c_sex_id', 'c_year_id', 'x_0', 'x_1', 'x_2',
        'integrand_name', 'minimum_meas_cv', 'rate'
    ])


def test_format_fit(ihme, dismod, df):
    d = DismodExtractor(path=Path('temp.db'))
    pred = d.format_predictions_for_ihme(gbd_round_id=6, samples=False)
    # This prediction data frame is longer for each demographic group than the prediction
    # dataframe in `test_get_predictions` because there is an extra measure.
    assert len(pred) == 528
    assert all(pred.columns == [
        'location_id', 'year_id', 'age_group_id', 'sex_id', 'measure_id', 'mean'
    ])

if __name__ == '__main__':
    os.chdir('/Users/gma/Projects/IHME/GIT/cascade-at')
    test_empty_database()
    args = (None, None, None)
    test_run_dismod_fit_predict(*args)
    test_get_predictions(*args)
    test_format_fit(*args)
