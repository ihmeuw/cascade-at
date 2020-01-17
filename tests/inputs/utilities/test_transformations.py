import numpy as np
from cascade_at.inputs.utilities.transformations import COVARIATE_TRANSFORMS


def test_identity():
    assert COVARIATE_TRANSFORMS[0](1) == 1


def test_log():
    assert COVARIATE_TRANSFORMS[1](1) == 0


def test_logit():
    assert COVARIATE_TRANSFORMS[2](1) == np.inf


def test_squared():
    assert COVARIATE_TRANSFORMS[3](1) == 1
    assert COVARIATE_TRANSFORMS[3](2) == 4
    assert COVARIATE_TRANSFORMS[3](3) == 9


def test_sqrt():
    assert COVARIATE_TRANSFORMS[4](1) == 1
    assert COVARIATE_TRANSFORMS[4](4) == 2
    assert COVARIATE_TRANSFORMS[4](9) == 3


def test_scale1000():
    assert COVARIATE_TRANSFORMS[5](1) == 1000
