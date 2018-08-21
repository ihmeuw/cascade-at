import pytest

from cascade.model.priors import (
    GaussianPrior,
    UniformPrior,
    LaplacePrior,
    StudentsTPrior,
    LogGaussianPrior,
    LogLaplacePrior,
    LogStudentsTPrior,
)


def test_happy_construction():
    UniformPrior(0, -1, 1, "test")
    GaussianPrior(0, 1, -10, 10, "test2")
    LaplacePrior(0, 1, -10, 10, "test3")
    StudentsTPrior(0, 1, 0.5, -10, 10, "test4")
    LogGaussianPrior(0, 1, 0.5, -10, 10, "test5")
    LogLaplacePrior(0, 1, 0.5, -10, 10, "test6")
    LogStudentsTPrior(0, 1, 0.5, 0.5, -10, 10, "test7")


def test_prior_equality():
    a = GaussianPrior(0, 1)
    b = GaussianPrior(0, 1)
    assert a == b

    a = GaussianPrior(0, 1, -1, 1)
    b = GaussianPrior(0, 1, -1, 1)
    assert a == b

    a = UniformPrior(10)
    b = UniformPrior(10)
    assert a == b

    a = UniformPrior(10, name="test_prior")
    b = UniformPrior(10, name="test_prior")
    assert a == b


def test_prior_nonequality():
    a = GaussianPrior(0, 1)
    b = GaussianPrior(1, 1)
    assert a != b

    a = UniformPrior(1)
    b = UniformPrior(-1)
    assert a != b

    a = GaussianPrior(0, 1, name="test_prior")
    b = GaussianPrior(0, 1, name="other_test_prior")
    assert a != b

    a = GaussianPrior(0, 1)
    b = UniformPrior(0)
    assert a != b


def test_prior_hashing():
    s = {GaussianPrior(0, 1), UniformPrior(1), GaussianPrior(0, 1), UniformPrior(2), UniformPrior(1)}

    assert len(s) == 3
    assert GaussianPrior(0, 1) in s
    assert UniformPrior(10) not in s


def test_bounds_check():
    with pytest.raises(ValueError) as excinfo:
        UniformPrior(0, 1, 1)
    assert "Bounds are inconsistent" in str(excinfo.value)


def test_validate_standard_deviation():
    with pytest.raises(ValueError) as excinfo:
        GaussianPrior(0, -1)
    assert "must be positive" in str(excinfo.value)


def test_validate_nu():
    with pytest.raises(ValueError) as excinfo:
        StudentsTPrior(0, 1, -1)
    assert "must be positive" in str(excinfo.value)
