import pytest

from cascade_at.inputs.demographics import Demographics


@pytest.fixture(scope='module')
def D(ihme):
    d = Demographics(gbd_round_id=6)
    return d


@pytest.fixture(scope='module')
def D_with_loc_set(ihme):
    d = Demographics(gbd_round_id=6, location_set_version_id=544)
    return d


@pytest.mark.parametrize("attribute", [
    'age_group_id', 'location_id', 'year_id', 'sex_id'
])
def test_demographics(D, attribute):
    assert type(getattr(D, attribute)) == list
    if not attribute == 'location_id':
        assert getattr(D, attribute)


def test_demographics_with_location_set(D_with_loc_set):
    assert type(getattr(D_with_loc_set, 'location_id')) == list
    assert getattr(D_with_loc_set, 'location_id')
