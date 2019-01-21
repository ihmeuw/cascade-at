from cascade.model.age_time_grid import AgeTimeGrid


def test_create():
    cols = ["var_id"]
    atg = AgeTimeGrid(([0, 1, 10], [2000, 2010]), cols)

    assert len(atg.grid) == 6
    assert len(atg.mulstd) == 3

    cols = ["var_id", "other_id", "residual"]
    atg = AgeTimeGrid(([0, 1, 10], [2000, 2010]), cols)
    assert not (set(cols) - set(atg.grid.columns))


def test_assign():
    cols = ["var_id"]
    atg = AgeTimeGrid(([0, 1, 10], [2000, 2010]), cols)

    assert len(atg.grid) == 6
    assert len(atg.mulstd) == 3
    atg[1, 2010] = 37
    assert int(atg[1, 2010].var_id) == 37
