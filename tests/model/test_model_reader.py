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
