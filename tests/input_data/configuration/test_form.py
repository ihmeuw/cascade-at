from cascade.core.form import Form
from cascade.input_data.configuration.form import SmoothingPrior, Eta, StudentsDOF
from cascade.model import priors


class DummyForm(Form):
    eta = Eta()
    students_dof = StudentsDOF()
    log_students_dof = StudentsDOF()

    prior = SmoothingPrior()


def test_SmoothingPrior__full_form_validation__success():
    f = DummyForm({"prior": {"prior_type": "value", "density": "uniform", "min": 0, "max": 10}})
    assert not f.validate_and_normalize()
    assert f.prior.prior_object == priors.Uniform(0, 10)

    f = DummyForm({"prior": {"prior_type": "value", "density": "log_gaussian", "mean": 0, "std": 1, "eta": 1}})
    assert not f.validate_and_normalize()
    assert f.prior.prior_object == priors.LogGaussian(0, 1, 1)


def test_SmoothingPrior__full_form_validation__fail():
    f = DummyForm({"prior": {"prior_type": "value", "density": "uniform", "min": 10, "max": 0}})
    assert f.validate_and_normalize() == [
        (
            "prior",
            "prior",
            "Parameters incompatible with density 'uniform': Bounds are inconsistent: lower=10.0 mean=5.0 upper=0.0",
        )
    ]

    f = DummyForm({"prior": {"prior_type": "value", "density": "gaussian", "mean": 0, "std": -1}})
    assert f.validate_and_normalize() == [
        (
            "prior",
            "prior",
            "Parameters incompatible with density 'gaussian':"
            " Standard deviation must be positive: standard deviation=-1.0",
        )
    ]


def test_SmoothingPrior__global_eta():
    # Non-log distributions shouldn't get the global eta
    f = DummyForm(
        {
            "eta": {"priors": 0.001, "data": 0.001},
            "prior": {"prior_type": "value", "density": "uniform", "min": 0, "max": 10},
        }
    )
    f.validate_and_normalize()
    assert f.prior.prior_object == priors.Uniform(0, 10)

    # But log distributions which don't have their own eta should
    f = DummyForm(
        {
            "eta": {"priors": 0.001, "data": 0.001},
            "prior": {"prior_type": "value", "density": "log_gaussian", "mean": 0, "std": 0.01},
        }
    )
    f.validate_and_normalize()
    assert f.prior.prior_object == priors.LogGaussian(0, 0.01, eta=0.001)

    # But if they do have their own eta, they shouldn't
    f = DummyForm(
        {
            "eta": {"priors": 0.001, "data": 0.001},
            "prior": {"prior_type": "value", "density": "log_gaussian", "mean": 0, "std": 0.01, "eta": 0.002},
        }
    )
    f.validate_and_normalize()
    assert f.prior.prior_object == priors.LogGaussian(0, 0.01, eta=0.002)


def test_SmoothingPrior__global_nu():
    # a students distribution, which gets its nu from the global default
    f = DummyForm(
        {
            "students_dof": {"priors": 5, "data": 5},
            "prior": {"prior_type": "value", "density": "students", "mean": 0, "std": 0.1},
        }
    )
    f.validate_and_normalize()
    assert f.prior.prior_object == priors.StudentsT(0, 0.1, 5)

    # a students distribution, who's local nu overrides the global default
    f = DummyForm(
        {
            "students_dof": {"priors": 5, "data": 5},
            "prior": {"prior_type": "value", "density": "students", "mean": 0, "std": 0.1, "nu": 3},
        }
    )
    f.validate_and_normalize()
    assert f.prior.prior_object == priors.StudentsT(0, 0.1, 3)

    # a log students distribution which is missing nu because the global default is for non-log students
    f = DummyForm(
        {
            "students_dof": {"priors": 5, "data": 5},
            "prior": {"prior_type": "value", "density": "log_students", "mean": 0, "std": 0.1},
        }
    )
    assert set(f.validate_and_normalize()) == {
        ("prior", "prior", "Parameters incompatible with density 'log_students': Nu must be greater than 2: nu=None")
    }

    # a students distribution which is missing nu because the global default is for log students
    f = DummyForm(
        {
            "log_students_dof": {"priors": 5, "data": 5},
            "prior": {"prior_type": "value", "density": "students", "mean": 0, "std": 0.1},
        }
    )
    assert set(f.validate_and_normalize()) == {
        ("prior", "prior", "Parameters incompatible with density 'students': Nu must be greater than 2: nu=None")
    }
