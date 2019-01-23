import numpy as np
from numpy import isclose, nan, isnan
import pandas as pd

from cascade.model.age_time_grid import AgeTimeGrid
from cascade.model.model_reader import (
    _read_vars_one_field, _read_residuals_one_field, _samples_one_field, _construct_var_id_from_var_table
)


def test_read_vars_one_field():
    table = pd.DataFrame(dict(
        scale_var_id=np.arange(6),
        scale_var_value=np.linspace(0, 1, 6),
    ))

    id_draw = AgeTimeGrid([50], [1995, 2015], "var_id")
    id_draw[50, 1995] = [4]
    id_draw[50, 2015] = [5]

    var_out = _read_vars_one_field(table, id_draw, "scale_var")
    grid = var_out.grid
    assert set(grid.columns) == {"age", "time", "mean"}
    assert len(grid) == 2
    assert isclose(float(grid.loc[grid.time == 1995, "mean"]), 0.8)
    assert isclose(float(grid.loc[grid.time == 2015, "mean"]), 1.0)


def test_read_vars_one_field_mulstd():
    table = pd.DataFrame(dict(
        scale_var_id=np.arange(6),
        scale_var_value=np.linspace(0, 1, 6),
    ))

    id_draw = AgeTimeGrid([50], [1995, 2015], "var_id")
    id_draw[50, 1995] = [4]
    id_draw[50, 2015] = [5]
    id_draw.mulstd["value"].loc[:, "var_id"] = 3

    var_out = _read_vars_one_field(table, id_draw, "scale_var")

    assert isclose(var_out.get_mulstd("value"), 0.6)


def test_read_residuals_one_field():
    table = pd.DataFrame(dict(
        fit_var_id=np.arange(6),
        fit_var_value=np.linspace(0, 1, 6),
        residual_value=10 + np.linspace(0, 1, 6),
        residual_dage=20 + np.linspace(0, 1, 6),
        residual_dtime=30 + np.linspace(0, 1, 6),
        lagrange_value=40 + np.linspace(0, 1, 6),
        lagrange_dage=50 + np.linspace(0, 1, 6),
        lagrange_dtime=60 + np.linspace(0, 1, 6),
    ))

    id_draw = AgeTimeGrid([50], [1995, 2015], "var_id")
    id_draw[50, 1995] = [2]
    id_draw[50, 2015] = [3]
    id_draw.mulstd["dage"].loc[:, "var_id"] = 4
    id_draw.mulstd["dtime"].loc[:, "var_id"] = 5

    var_out = _read_residuals_one_field(table, id_draw)
    grid = var_out.grid
    assert not (set(table.columns) - {"fit_var_id"} - set(grid.columns))
    assert len(grid) == 2
    for year, value in [(1995, 0.4), (2015, 0.6)]:
        assert isclose(float(grid.loc[grid.time == year, "fit_var_value"]), value)
        assert isclose(float(grid.loc[grid.time == year, "residual_value"]), 10 + value)
        assert isclose(float(grid.loc[grid.time == year, "residual_dage"]), 20 + value)
        assert isclose(float(grid.loc[grid.time == year, "residual_dtime"]), 30 + value)
        assert isclose(float(grid.loc[grid.time == year, "lagrange_value"]), 40 + value)
        assert isclose(float(grid.loc[grid.time == year, "lagrange_dage"]), 50 + value)
        assert isclose(float(grid.loc[grid.time == year, "lagrange_dtime"]), 60 + value)

    assert isclose(float(var_out.mulstd["dage"].loc[:, "residual_value"]), 10.8)


def test_read_samples_one_field_mulstd():
    id_draw = AgeTimeGrid([50], [2004, 2005], "var_id")
    id_draw[50, 2004] = [4]  # var 4 is 2004
    id_draw[50, 2005] = [5]  # var 5 is 2005
    id_draw.mulstd["dtime"].loc[:, "var_id"] = 3

    var_cnt = 6
    sample_cnt = 5
    sample_index = np.repeat(np.arange(0, sample_cnt), var_cnt)
    var_id = np.tile(np.arange(0, var_cnt), sample_cnt)
    table = pd.DataFrame(dict(
        sample_id=np.arange(0, var_cnt * sample_cnt),
        sample_index=sample_index,
        var_id=var_id,
        var_value=(2000 + var_id + sample_index) * 0.001,
    ))

    var_out = _samples_one_field(table, id_draw)

    for a, t in var_out.age_time():
        samples = var_out[a, t]
        assert len(samples) == sample_cnt
        samples = samples.sort_values("mean")
        for idx in range(sample_cnt):
            assert isclose(samples.iloc[idx]["mean"], (t + idx) * 0.001)
    mulstd_out = var_out.mulstd["dtime"]
    assert len(mulstd_out) == sample_cnt
    for idx in range(sample_cnt):
        assert isclose(mulstd_out[mulstd_out.idx == idx]["mean"], (2003 + idx) * 0.001)


def test_add_one_field_to_vars():
    sub_grid_df = pd.DataFrame(dict(
        var_id=[4, 5, 6],
        var_type=["rate", "rate", "mulstd_value"],
        smooth_id=0,
        age_id=0,
        time_id=[0, 2, nan],
        node_id=0,
        rate_id=1,
        integrand_id=nan,
        covariate_id=nan,
    ))
    age = pd.DataFrame(dict(age_id=[0, 1, 2], age=[0, 50, 100]))
    time = pd.DataFrame(dict(time_id=[0, 1, 2], time=[2000, 2005, 2010]))
    var = _construct_var_id_from_var_table(sub_grid_df, age, time)
    assert var is not None
    assert int(var[0, 2000].var_id) == 4
    assert int(var[0, 2010].var_id) == 5
    assert int(var.mulstd["value"].at[0, "var_id"]) == 6
    assert isnan(var.mulstd["dage"].at[0, "var_id"])
