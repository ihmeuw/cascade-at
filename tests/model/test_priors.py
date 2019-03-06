import pytest
import numpy as np
from numpy import isclose
from numpy.random import RandomState

from cascade.model.priors import (
    Constant,
    Gaussian,
    Uniform,
    Laplace,
    StudentsT,
    LogGaussian,
    LogLaplace,
    LogStudentsT,
    PriorError,
)


def test_happy_construction():
    Uniform(-1, 1, 0, name="test")
    Uniform(-1, 1, 0, 0.5, name="test")
    Gaussian(0, 1, -10, 10, name="test2")
    Gaussian(0, 1, -10, 10, 0.5, name="test2")
    Laplace(0, 1, -10, 10, name="test3")
    Laplace(0, 1, -10, 10, 0.5, name="test3")
    StudentsT(0, 1, 2.5, -10, 10, name="test4")
    LogGaussian(0, 1, 0.5, -10, 10, name="test5")
    LogLaplace(0, 1, 0.5, -10, 10, name="test6")
    LogStudentsT(0, 1, 2.5, 0.5, -10, 10, name="test7")


def test_prior_equality():
    a = Gaussian(0, 1)
    b = Gaussian(0, 1)
    assert a == b

    a = Gaussian(0, 1, -1, 1)
    b = Gaussian(0, 1, -1, 1)
    assert a == b

    a = Uniform(0, 10)
    b = Uniform(0, 10)
    assert a == b

    a = Uniform(0, 10, name="test_prior")
    b = Uniform(0, 10, name="test_prior")
    assert a == b


def test_prior_nonequality():
    a = Gaussian(0, 1)
    b = Gaussian(1, 1)
    assert a != b

    a = Uniform(0, 1)
    b = Uniform(-1, 0)
    assert a != b

    a = Gaussian(0, 1, name="test_prior")
    b = Gaussian(0, 1, name="other_test_prior")
    assert a != b

    a = Gaussian(0, 1)
    b = Uniform(0, 1)
    assert a != b


def test_prior_sort():
    priors = [
        Uniform(lower=1e-10, upper=1, mean=5e-5, name="iota"),
        Gaussian(0, 1, name="other_test_prior"),
        Uniform(0, 1),
    ]

    # NOTE: This is a weak test of actual sorting behavior however all I
    # actually care about is that the sort is stable, I don't really care
    # what the order is
    assert sorted(priors) == sorted(reversed(priors))


def test_prior_hashing():
    s = {Gaussian(0, 1), Uniform(0, 1), Gaussian(0, 1), Uniform(0, 2), Uniform(0, 1)}

    assert len(s) == 3
    assert Gaussian(0, 1) in s
    assert Uniform(0, 10) not in s


def test_prior_hashing__near_miss():
    assert hash(Gaussian(0, 1.0000000000000001)) == hash(Gaussian(0, 1))
    assert hash(Gaussian(0, 1.000000000000001)) != hash(Gaussian(0, 1))


def test_bounds_check():
    with pytest.raises(PriorError) as excinfo:
        Uniform(0, -1, 1)
    assert "Bounds are inconsistent" in str(excinfo.value)


def test_validate_standard_deviation():
    with pytest.raises(PriorError) as excinfo:
        Gaussian(0, -1)
    assert "must be positive" in str(excinfo.value)


@pytest.mark.parametrize("bad_nu", [-1, -3, 0, 2, 1.99])
def test_validate_nu(bad_nu):
    with pytest.raises(PriorError) as excinfo:
        StudentsT(0, 1, bad_nu)
    assert "must be greater" in str(excinfo.value)


@pytest.fixture
def rng():
    return RandomState(34257234)


def test_const_fit():
    """A constant distribution is unchanged."""
    dist = Constant(0.023)
    assert isclose(dist.rvs(), 0.023)
    assert isclose(dist.mle([6, 24, 327]).mean, 0.023)


def test_uniform_fit(rng):
    dist = Uniform(-0.4, 0.6, 0.5)
    draws = dist.rvs(size=10000, random_state=rng)
    new_dist = dist.mle(draws)
    assert isclose(new_dist.mean, 0.1, atol=0.01)


@pytest.mark.parametrize("cls,params", [
    (Gaussian, (0.1, 1, -10, 10)),
    (Gaussian, (0.1, 1, 0, 0.2)),
    (Laplace, (0, 1, -10, 10)),
    (StudentsT, (0, 1, 2.7, -10, 10)),
])
def test_mle(cls, params, rng):
    dist = cls(*params)
    draw_dist = dist
    if hasattr(dist, "mean"):
        draw_dist = draw_dist.assign(mean=0.1)
    if hasattr(dist, "standard_deviation"):
        draw_dist = draw_dist.assign(standard_deviation=0.04)

    draws = draw_dist.rvs(size=10000, random_state=rng)
    assert np.all((dist.lower <= draws) & (draws <= dist.upper))
    new_dist = dist.mle(draws)

    if hasattr(dist, "mean"):
        assert isclose(new_dist.mean, 0.1, rtol=0.2)

    if hasattr(dist, "standard_deviation"):
        assert isclose(new_dist.standard_deviation, 0.04, rtol=0.2)
