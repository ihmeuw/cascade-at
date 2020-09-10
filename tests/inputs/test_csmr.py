import pytest

from cascade_at.dismod.constants import IntegrandEnum
from cascade_at.inputs.csmr import get_best_cod_correct


@pytest.mark.parametrize("hold_out", [0, 1])
def test_configure_for_dismod(csmr, hold_out):
    assert csmr.configure_for_dismod(hold_out=hold_out)['hold_out'].iloc[0] == hold_out


@pytest.fixture(scope='module')
def df_for_dismod(csmr):
    return csmr.configure_for_dismod(hold_out=0)


@pytest.mark.parametrize("column,value", [
    ("age_lower", 0.),
    ("age_upper", 0.01917808),
    ("age_group_id", 2),
    ("location_id", 70),
    ("time_lower", 1990),
    ("time_upper", 1991),
    ("measure", IntegrandEnum.mtspecific.name),
    ("meas_value", 5e-06),
    ("sex_id", 2),
    ("hold_out", 0)
])
def test_all_columns(df_for_dismod, column, value):
    assert df_for_dismod[column].iloc[0] == value


def test_columns(csmr, df_for_dismod):
    assert all([c in csmr.columns_to_keep for c in df_for_dismod.columns])


def test_get_best_codcorrect(ihme):
    # GBD round 6 should be fixed
    assert get_best_cod_correct(6) == 14770
