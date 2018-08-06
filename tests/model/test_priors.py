from cascade.model.priors import GaussianPrior, UniformPrior


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

    a = UniformPrior(10, "test_prior")
    b = UniformPrior(10, "test_prior")
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
