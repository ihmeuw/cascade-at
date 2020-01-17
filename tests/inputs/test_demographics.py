import pytest

from cascade_at.inputs.demographics import Demographics


@pytest.fixture(scope='module')
def D(ihme):
    d = Demographics(gbd_round_id=6)
    return d


@pytest.mark.parametrize("attribute", [
    'age_group_id', 'location_id', 'year_id', 'sex_id'
])
def test_demographics(D, attribute):
    assert type(getattr(D, attribute)) == list
    assert getattr(D, attribute)
