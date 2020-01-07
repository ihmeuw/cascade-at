import pytest

from cascade_at.inputs.demographics import Demographics


@pytest.mark.parametrize("attribute", [
    'age_group_id', 'location_id', 'year_id', 'sex_id'
])
def test_demographics(ihme, attribute):
    D = Demographics(gbd_round_id=6)
    assert type(getattr(D, attribute)) == list
    assert getattr(D, attribute)
