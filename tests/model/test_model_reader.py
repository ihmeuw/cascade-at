import numpy as np
from numpy import isclose
import pandas as pd

from cascade.model.age_time_grid import AgeTimeGrid
from cascade.model.model_reader import _read_vars_one_field, _read_residuals_one_field


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
    assert set(grid.columns) == {"age", "time", "mean", "idx"}
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
