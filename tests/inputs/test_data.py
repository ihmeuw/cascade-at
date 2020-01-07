import pytest
import copy

from cascade_at.inputs.utilities.transformations import RELABEL_INCIDENCE_MAP


@pytest.mark.parametrize("relabel_incidence", [1, 2, 3])
def test_relabel_incidence(cv, relabel_incidence):
    df = cv.configure_for_dismod(relabel_incidence=relabel_incidence)
    assert df['measure'].iloc[0] == RELABEL_INCIDENCE_MAP[relabel_incidence]['incidence']


@pytest.mark.parametrize("measures_to_exclude,relabel_incidence,hold_out", [
    (None, 1, 0),
    (['incidence'], 1, 0),
    (['Sincidence'], 1, 1)
])
def test_measures_exclude(cv, measures_to_exclude, relabel_incidence, hold_out):
    df = cv.configure_for_dismod(measures_to_exclude=measures_to_exclude, relabel_incidence=relabel_incidence)
    assert df['hold_out'].iloc[0] == hold_out


@pytest.fixture
def df_for_dismod(cv):
    return cv.configure_for_dismod(measures_to_exclude=None, relabel_incidence=2)


@pytest.mark.parametrize("column,value", [
    ("location_id", 101),
    ("time_lower", 1990),
    ("time_upper", 1991),
    ("sex_id", 2),
    ("measure", "Tincidence"),
    ("meas_value", 4e-05),
    ("meas_std", 3e-06),
    ("age_lower", 0.0),
    ("age_upper", 15.0),
    ("hold_out", 0),
    ("name", "342686")
])
def test_all_columns(df_for_dismod, column, value):
    assert df_for_dismod[column].iloc[0] == value


def test_outside_year_bounds(cv):
    cs = copy.deepcopy(cv)
    cs.raw.year_start = 1980
    assert cs.configure_for_dismod(relabel_incidence=1).empty


def test_outside_locations(cv):
    cs = copy.deepcopy(cv)
    cs.demographics.location_id = [160]
    assert cs.configure_for_dismod(relabel_incidence=1).empty


def test_exclude_outliers(cv):
    cs = copy.deepcopy(cv)
    cs.raw.is_outlier = 1
    assert cs.configure_for_dismod(relabel_incidence=1).empty


def test_columns(cv, df_for_dismod):
    assert all([c in cv.columns_to_keep for c in df_for_dismod.columns])
