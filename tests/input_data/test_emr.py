import pytest

import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal

from cascade.dismod.constants import DensityEnum
from cascade.executor.execution_context import make_execution_context
from cascade.stats import meas_bounds_to_stdev

from cascade.input_data.emr import (
    _emr_from_sex_and_node_specific_csmr_and_prevalence,
    _make_interpolators,
    _prepare_csmr,
    _collapse_times,
    _collapse_ages_unweighted,
    _collapse_ages_weighted,
)

POINT_PREVALENCE = pd.DataFrame(
    {
        "age_lower": [0, 1, 10, 15, 20] * 2,
        "age_upper": [0, 1, 10, 15, 20] * 2,
        "time_lower": [1990] * 5 + [1995] * 5,
        "time_upper": [1990] * 5 + [1995] * 5,
        "sex_id": [3] * 5 * 2,
        "node_id": [6] * 5 * 2,
        "density": [DensityEnum.gaussian] * 5 * 2,
        "weight": ["constant"] * 5 * 2,
        "mean": [0.01, 0.02, 0.03, 0.04, 0.05] * 2,
        "standard_error": [0.005, 0.004, 0.003, 0.002, 0.001] * 2,
        "measure": ["prevalence"] * 5 * 2,
    }
)

POINT_CSMR = pd.DataFrame(
    {
        "age": [0, 1, 10, 15, 20] * 2,
        "age_lower": [0, 1, 10, 15, 20] * 2,
        "age_upper": [0, 1, 10, 15, 20] * 2,
        "time": [1990] * 5 + [1995] * 5,
        "time_lower": [1990] * 5 + [1995] * 5,
        "time_upper": [1990] * 5 + [1995] * 5,
        "sex_id": [3] * 5 * 2,
        "node_id": [6] * 5 * 2,
        "mean": [0.006, 0.007, 0.008, 0.009, 0.01] * 2,
        "standard_error": [0.0005, 0.0004, 0.0003, 0.0002, 0.0001] * 2,
    }
)

SPAN_PREVALENCE = pd.DataFrame(
    {
        "age_lower": [0, 3, 20, 50, 70, 80],
        "age_upper": [0.001, 4, 25, 55, 90, 80],
        "time_lower": [1990, 1995, 2000, 2005, 2010, 2015],
        "time_upper": [1995, 2000, 2005, 2010, 2015, 2020],
        "sex_id": [3] * 6,
        "node_id": [6] * 6,
        "density": [DensityEnum.gaussian] * 6,
        "weight": ["constant"] * 6,
        "mean": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "standard_error": [0.005, 0.004, 0.003, 0.002, 0.001, 0.001],
        "measure": ["prevalence"] * 6,
    }
)


@pytest.fixture(scope="module")
def csmr_surface():
    a = -0.01
    b = -0.02
    c = 1.0
    csmr_mean = lambda age, time: -(-a * 0 - b * 1980 - c * 0.001 + a * age + b * time) / c

    csmr_rows = []
    for age in range(0, 120):
        for time in range(1980, 2040):
            csmr_rows.append(
                {
                    "age": age,
                    "time": time,
                    "sex_id": 3,
                    "node_id": 6,
                    "mean": csmr_mean(age, time),
                    "standard_error": 0.01,
                }
            )

    return pd.DataFrame(csmr_rows), csmr_mean


def test_interpolators__points_only(csmr_surface):
    csmr, source = csmr_surface
    interps, _ = _make_interpolators(csmr)

    for age in np.linspace(0, 100, 200):
        for time in np.linspace(1990, 2018, 50):
            assert np.isclose(interps["both"](age, time), source(age, time))


def test_interpolators__across_ages(csmr_surface):
    csmr, source = csmr_surface
    interps, stderr_interp = _make_interpolators(csmr)

    for age in np.linspace(0, 100, 200):
        mean = np.mean([source(age, time) for time in np.linspace(1980, 2040, 100)])
        assert abs(interps["age"](age) - mean) < stderr_interp["age"](age) * 2


def test_interpolators__across_times(csmr_surface):
    csmr, source = csmr_surface
    interps, stderr_interp = _make_interpolators(csmr)

    for time in np.linspace(1990, 2018, 50):
        mean = np.mean([source(age, time) for age in np.linspace(0, 120, 200)])
        assert abs(interps["time"](time) - mean) < stderr_interp["time"](time) * 2


def test_emr_from_sex_and_node_specific_csmr_and_prevalence__perfect_alignment__points_only():
    emr = _emr_from_sex_and_node_specific_csmr_and_prevalence(POINT_CSMR, POINT_PREVALENCE)

    emr = emr.set_index(["age_lower", "age_upper", "time_lower", "time_upper", "sex_id", "node_id"])
    csmr = POINT_CSMR.set_index(["age_lower", "age_upper", "time_lower", "time_upper", "sex_id", "node_id"])
    prevalence = POINT_PREVALENCE.set_index(["age_lower", "age_upper", "time_lower", "time_upper", "sex_id", "node_id"])

    assert len(emr) == len(prevalence)

    assert np.allclose(emr["mean"], csmr["mean"] / prevalence["mean"])


def test_emr_from_sex_and_node_specific_csmr_and_prevalence__perfect_alignment__spans(csmr_surface):
    csmr, source = csmr_surface

    emr = _emr_from_sex_and_node_specific_csmr_and_prevalence(csmr, SPAN_PREVALENCE)

    assert len(emr) == len(SPAN_PREVALENCE)

    for (_, pr), (_, er) in zip(SPAN_PREVALENCE.iterrows(), emr.iterrows()):
        pmean = pr["mean"]
        cmean = source((pr["age_lower"] + pr["age_upper"]) / 2, (pr["time_lower"] + pr["time_upper"]) / 2)
        assert er["mean"] - cmean / pmean < er["standard_error"]


def test_collapse_times():
    df = pd.DataFrame({"time_lower": [1990, 1990, 1990], "time_upper": [1990, 1995, 2000]})

    df_new = _collapse_times(df)

    assert all(df_new["time"] == [1990, 1992.5, 1995])


def test_collapse_ages_unweighted():
    df = pd.DataFrame({"age_lower": [0, 1, 3, 53, 100.5, 1000000], "age_upper": [0.01, 2, 4.5, 57, 101, 1000005]})

    df_new = _collapse_ages_unweighted(df)

    assert all(df_new["age"] == [0.005, 1.5, 3.75, 55, 100.75, 1000002.5])


def test_collapse_ages_weighted(mocker):
    mock_get_life_table = mocker.patch("cascade.input_data.db.demographics.db_queries").get_life_table
    mock_get_life_table.return_value = pd.DataFrame(
        {
            "age_group_id": [1, 2, 3, 4, 5, 6],
            "location_id": [0] * 6,
            "year_id": [1990] * 6,
            "sex_id": [1, 2, 3, 1, 2, 3],
            "mean": [0.008, 0.5, 1, 3, 0.1, 2],
        }
    )

    df = pd.DataFrame(
        {
            "node_id": [0, 0, 0, 0, 0, 0],
            "sex_id": [1, 2, 3, 1, 2, 3],
            "time_lower": [1990, 1990, 1990, 1990, 1990, 1990],
            "age_group_id": [1, 2, 3, 4, 5, 6],
            "age_lower": [0, 1, 3, 53, 100.5, 1000000],
            "age_upper": [0.01, 2, 4.5, 57, 101, 1000005],
        }
    )

    df_new = _collapse_ages_weighted(mocker.Mock(), df)

    assert all(df_new["age"] == [0.008, 1.5, 4, 56, 100.6, 1000002])


def test_prepare_csmr(mocker):
    csmr = pd.DataFrame(
        {
            "age_lower": [0, 1, 10, 15, 20] * 2,
            "age_upper": [0, 1, 10, 15, 20] * 2,
            "time_lower": [1990] * 5 + [1995] * 5,
            "time_upper": [1990] * 5 + [1995] * 5,
            "sex_id": [3] * 5 * 2,
            "location_id": [6] * 5 * 2,
            "meas_value": [0.006, 0.007, 0.008, 0.009, 0.01] * 2,
            "meas_lower": [0.006, 0.007, 0.008, 0.009, 0.01] * 2,
            "meas_upper": [0.006, 0.007, 0.008, 0.009, 0.01] * 2,
        }
    )
    mock_age_groups_to_ranges = mocker.patch("cascade.input_data.emr.age_groups_to_ranges")
    mock_age_groups_to_ranges.side_effect = lambda ec, df, keep_age_group_id: df
    ec = make_execution_context()

    csmr_new = _prepare_csmr(ec, csmr, False)

    assert all(csmr_new["node_id"] == csmr["location_id"])
    assert np.allclose(csmr_new["mean"], csmr["meas_value"])
    assert all(csmr_new["time"] == csmr["time_lower"])
    assert_frame_equal(meas_bounds_to_stdev(csmr)[["mean", "standard_error"]], csmr_new[["mean", "standard_error"]])
