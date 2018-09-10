import pytest

from cascade.model.priors import (
    GaussianPrior,
    UniformPrior,
    LaplacePrior,
    StudentsTPrior,
    LogGaussianPrior,
    LogLaplacePrior,
    LogStudentsTPrior,
    PriorError,
)


def test_happy_construction():
    UniformPrior(-1, 1, 0, name="test")
    UniformPrior(-1, 1, 0, 0.5, name="test")
    GaussianPrior(0, 1, -10, 10, name="test2")
    GaussianPrior(0, 1, -10, 10, 0.5, name="test2")
    LaplacePrior(0, 1, -10, 10, name="test3")
    LaplacePrior(0, 1, -10, 10, 0.5, name="test3")
    StudentsTPrior(0, 1, 0.5, -10, 10, name="test4")
    LogGaussianPrior(0, 1, 0.5, -10, 10, name="test5")
    LogLaplacePrior(0, 1, 0.5, -10, 10, name="test6")
    LogStudentsTPrior(0, 1, 0.5, 0.5, -10, 10, name="test7")


def test_prior_equality():
    a = GaussianPrior(0, 1)
    b = GaussianPrior(0, 1)
    assert a == b

    a = GaussianPrior(0, 1, -1, 1)
    b = GaussianPrior(0, 1, -1, 1)
    assert a == b

    a = UniformPrior(0, 10)
    b = UniformPrior(0, 10)
    assert a == b

    a = UniformPrior(0, 10, name="test_prior")
    b = UniformPrior(0, 10, name="test_prior")
    assert a == b


def test_prior_nonequality():
    a = GaussianPrior(0, 1)
    b = GaussianPrior(1, 1)
    assert a != b

    a = UniformPrior(0, 1)
    b = UniformPrior(-1, 0)
    assert a != b

    a = GaussianPrior(0, 1, name="test_prior")
    b = GaussianPrior(0, 1, name="other_test_prior")
    assert a != b

    a = GaussianPrior(0, 1)
    b = UniformPrior(0, 1)
    assert a != b


def test_prior_sort():
    priors = [
        UniformPrior(lower=1e-10, upper=1, mean=5e-5, name="iota"),
        GaussianPrior(0, 1, name="other_test_prior"),
        UniformPrior(0, 1),
    ]

    # NOTE: This is a weak test of actual sorting behavior however all I
    # actually care about is that the sort is stable, I don't really care
    # what the order is
    assert sorted(priors) == sorted(reversed(priors))


def test_prior_hashing():
    s = {GaussianPrior(0, 1), UniformPrior(0, 1), GaussianPrior(0, 1), UniformPrior(0, 2), UniformPrior(0, 1)}

    assert len(s) == 3
    assert GaussianPrior(0, 1) in s
    assert UniformPrior(0, 10) not in s


def test_bounds_check():
    with pytest.raises(PriorError) as excinfo:
        UniformPrior(0, -1, 1)
    assert "Bounds are inconsistent" in str(excinfo.value)


def test_validate_standard_deviation():
    with pytest.raises(PriorError) as excinfo:
        GaussianPrior(0, -1)
    assert "must be positive" in str(excinfo.value)


def test_validate_nu():
    with pytest.raises(PriorError) as excinfo:
        StudentsTPrior(0, 1, -1)
    assert "must be positive" in str(excinfo.value)