"""
AgeTimeGrid isn't part of the API itself, but API members depend on it.
"""
import pytest

from numpy import isclose
import pandas as pd

from cascade.model.age_time_grid import AgeTimeGrid


def test_create():
    cols = ["var_id"]
    atg = AgeTimeGrid([0, 1, 10], [2000, 2010], cols)

    assert len(atg.grid) == 6
    assert len(atg.mulstd) == 3

    cols = ["var_id", "other_id", "residual"]
    atg = AgeTimeGrid([0, 1, 10], [2000, 2010], cols)
    assert not (set(cols) - set(atg.grid.columns))


def test_assign():
    cols = ["var_id"]
    atg = AgeTimeGrid([0, 1, 10], [2000, 2010], cols)

    assert len(atg.grid) == 6
    assert len(atg.mulstd) == 3
    atg[1, 2010] = 37
    assert int(atg[1, 2010].var_id) == 37


def test_create_one_age_time():
    atg0 = AgeTimeGrid(40, 2010, ["mean"])
    assert atg0 is not None


def test_create_wrong():
    atg = AgeTimeGrid([40], [2010], "not_a_column")
    assert "not_a_column" in atg.columns


def test_set_regions_single_column():
    cols = ["var_id"]
    atg = AgeTimeGrid([0, 1, 10], [2000, 2010], cols)
    with pytest.raises(TypeError):
        # You can't assign directly to a set of row elements.
        atg[:, :].var_id = [204]
    # You have to assign to columns instead, in order of atg.columns or with
    # a Pandas row.
    atg[:, :] = [204]
    # This returns a pd.Series
    assert (atg[10, 2000].var_id == 204).all()

    atg[5:17, :] = 37
    assert (atg[10, 2010].var_id == 37).all()
    assert (atg[1, 2010].var_id == 204).all()
    atg[:, 2005:2015] = [321]
    assert (atg[0, 2010].var_id == 321).all()
    assert (atg[0, 2000].var_id == 204).all()

    # You can use any iterable. This sets to 21.
    atg[:, 2005:2015] = {21: "hiya"}
    assert float(atg[0, 2010].var_id) == 21

    # Being outside the bounds is not OK, according to this.
    with pytest.raises(ValueError):
        atg[:, 2015:2020] = "won't even assign"
    # for a, t in atg.age_time():
    #     assert (atg[a, t].var_id != 24).all()


def test_multiple_columns():
    cols = ["height", "weight"]
    atg = AgeTimeGrid([3.7, 2.4, -15], [0, 5, 10], cols)
    for c in cols:
        assert c in atg.columns
    assert "mean" not in atg.columns
    atg[2.4, 10] = [6, 199]
    assert float(atg[2.4, 10].height) == 6
    assert float(atg[2.4, 10].weight) == 199
    with pytest.raises(ValueError):
        # You can't do this equality comparison.
        assert (atg[2.4, 10] == [6, 199]).all()
    atg[:, :] = [5.1, 130]
    assert float(atg[-15, 10].height) == 5.1

    # What if I go in and much with the ordering?
    # This could happen if we manipulate grid by hand, which we do.
    new_cols = list(atg.grid.columns)
    new_cols.remove("height")
    new_cols.remove("weight")
    new_cols.extend(["weight", "height"])
    atg.grid = atg.grid[new_cols]

    atg[3.7, 5] = [4.7, 205]
    assert float(atg[3.7, 5].height) == 4.7
    assert float(atg[3.7, 5].weight) == 205

    atg[2.4, 0].height = [5.6]
    assert isinstance(atg[2.4, 5], pd.DataFrame)


def test_mulstd():
    atg = AgeTimeGrid([50], [2000], ["clip"])
    value = atg.mulstd["value"]
    assert isinstance(value, pd.DataFrame)
    # This is the only way to assign.
    value.loc[:, "clip"] = 27
    assert value.iloc[0]["clip"] == 27
    assert (value.loc[:, "clip"] == 27).all()
    with pytest.raises(ValueError):
        assert value["clip"] == 27
    assert isclose(float(value["clip"]), 27)


def test_printing():
    atg = AgeTimeGrid([0, 10, 50], [2000, 2010], ["clip"])
    assert "variables" in str(atg)
    assert "2010" in repr(atg)
