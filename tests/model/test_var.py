from numpy import isclose, isnan
import pytest

from cascade.model.var import Var


def test_var_returns_a_float():
    var = Var([50, 17], [1990, 2000, 2010])
    var[:, :] = 34.7
    assert isinstance(var[50, 2000], float)

    var[17, 1990] = 204.3
    assert var[17, 1990] == 204.3


def test_works_as_a_function():
    const = Var([0], [0])
    const[0, 0] = 21.2
    assert const(0, 0) == 21.2
    assert const(365, -29) == 21.2


@pytest.mark.parametrize("a,t,v", [
    (0, 5, 36.9),
    (0, 10, 36.9),
    (1, 5, 36.9),
    (1, 10, 36.9),
    (29, 49, 36.9),
    (0.5, 7, 36.9),
])
def test_const_square(a, t, v):
    two = Var([0, 1], [5, 10])
    two[:, :] = 36.9
    assert isclose(two(a, t), v)


@pytest.mark.parametrize("a,t,v", [
    (1, 10, 6),
    (2, 10, 6),
])
def test_warp(a, t, v):
    warp = Var([0, 1], [5, 10, 15])
    warp[:, :] = 0.5
    warp[1, 10] = 6
    assert isclose(warp(a, t), v)
    assert warp(0.9, 10) > 5


@pytest.mark.parametrize("a,t,v", [
    (0, 2000, 0),
    (3, 2000, 0.6),
    (0, 200, 0),
    (3, 200, 0.6),
    (-7, 2000, 0),
    (24, 1975, 0.6),
])
def test_age_dimension(a, t, v):
    onea = Var([0, 1, 2, 3], [2000])
    for i in range(4):
        onea[i, :] = 0.2 * i
    assert isclose(onea(a, t), v)


@pytest.mark.parametrize("a,t,v", [
    (0, 2000, 0.6),
    (500, 2005, 0.9),
    (50, 1980, 0),
    (50, 2010, 0.9),
])
def test_time_dimension(a, t, v):
    times = [1990, 1995, 2000, 2005]
    onet = Var([50], times)
    for i in range(4):
        onet[50, times[i]] = 0.3 * i
    assert isclose(onet(a, t), v)


@pytest.mark.parametrize("name,value", [
    ("value", 3.7),
    ("dage", 2.4),
    ("dtime", -7.3),
    ("value", 4),  # Try an integer.
])
def test_mulstd(name, value):
    onet = Var([50, 60], [2000, 2010])
    onet.set_mulstd(name, value)
    assert onet.get_mulstd(name) == value

    onet.set_mulstd("dage", 2.4)
    onet.set_mulstd("dtime", -7.3)
    assert onet.get_mulstd("dage") == 2.4
    assert onet.get_mulstd("dtime") == -7.3


@pytest.mark.parametrize("name,value", [
    ("anything", 3.2),
    ("anything", 4),
    ("dage ", 3.2),
])
def test_mulstd_failure(name, value):
    onet = Var([50, 60], [2000, 2010])
    with pytest.raises(ValueError):
        onet.set_mulstd(name, value)


def test_mulstd_read_failure():
    onet = Var([50, 60], [2000, 2010])
    with pytest.raises(ValueError):
        onet.get_mulstd("else")

    # Here the key is good, but there is nothing there.
    assert isnan(onet.get_mulstd('dage'))
