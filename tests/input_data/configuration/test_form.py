from cascade.input_data.configuration.form import SmoothingPrior
from cascade.model import priors


def test_SmoothingPrior__full_form_validation__success():
    f = SmoothingPrior({"prior_type": "value", "density": "uniform", "min": 0, "max": 10})
    assert not f.validate_and_normalize()
    assert f.prior_object == priors.Uniform(0, 10)

    f = SmoothingPrior({"prior_type": "value", "density": "log_gaussian", "mean": 0, "std": 1, "eta": 1})
    assert not f.validate_and_normalize()
    assert f.prior_object == priors.LogGaussian(0, 1, 1)


def test_SmoothingPrior__full_form_validation__fail():
    f = SmoothingPrior({"prior_type": "value", "density": "uniform", "min": 10, "max": 0})
    assert f.validate_and_normalize() == [
        ("", "", "Parameters incompatible with density 'uniform': Bounds are inconsistent: lower=10.0 mean=5.0 upper=0.0")
    ]

    f = SmoothingPrior({"prior_type": "value", "density": "gaussian", "mean": 0, "std": -1})
    assert f.validate_and_normalize() == [
        (
            "",
            "",
            "Parameters incompatible with density 'gaussian':"
            " Standard deviation must be positive: standard deviation=-1.0",
        )
    ]
