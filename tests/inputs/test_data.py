import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

from cascade_at.inputs.data import CrosswalkVersion
from cascade_at.inputs.utilities.transformations import RELABEL_INCIDENCE_MAP


@pytest.fixture
def Demographics():
    d = SimpleNamespace()
    d.age_group_id = list(range(2, 12))
    d.location_id = [101]
    d.sex_id = [2]
    d.year_id = [1990, 1991]
    return d


@pytest.fixture
def cv(ihme, Demographics):
    cv = CrosswalkVersion(crosswalk_version_id=1, exclude_outliers=1,
                          demographics=Demographics, conn_def='dismod-at-dev')
    cv.raw = pd.DataFrame({
        'underlying_nid': np.nan,
        'nid': 230075,
        'field_citation_value': '',
        'source_type': '',
        'location_name': 'Canada',
        'location_id': 101,
        'sex': 'Female',
        'year_start': 1990,
        'year_end': 1991,
        'age_start': 0.0,
        'age_end': 15.0,
        'measure': 'incidence',
        'mean': 4e-05,
        'lower': 5e-05,
        'upper': 3e-05,
        'standard_error': 3e-06,
        'cases': 100.,
        'sample_size': 2000000.,
        'unit_type': '',
        'unit_value_as_published': 1,
        'uncertainty_type_value': 95,
        'representative_name': '',
        'urbanicity_type': '',
        'recall_type': '',
        'recall_type_value': np.nan,
        'sampling_type': '',
        'group': np.nan,
        'specificity': np.nan,
        'group_review': np.nan,
        'seq': 342686,
        'crosswalk_parent_seq': 321982,
        'variance': np.nan,
        'effective_sample_size': 2000000.,
        'design_effect': np.nan,
        'is_outlier': 0,
        'standardized.case.definition': '',
        'serum_plasma': np.nan,
        'orig_source': '',
        'age_split': 0,
        'standard_error_orig': 3e-06,
        'mean_orig': 4e-05,
        'input_type': '',
        'uncertainty_type': '',
        'underlying_field_citation_value': np.nan
    }, index=[0])
    return cv


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
    cv.raw.year_start = 1980
    assert cv.configure_for_dismod(relabel_incidence=1).empty


def test_outside_locations(cv):
    cv.demographics.location_id = [160]
    assert cv.configure_for_dismod(relabel_incidence=1).empty


def test_exclude_outliers(cv):
    cv.raw.is_outlier = 1
    assert cv.configure_for_dismod(relabel_incidence=1).empty


def test_columns(cv, df_for_dismod):
    assert all([c in cv.columns_to_keep for c in df_for_dismod.columns])
