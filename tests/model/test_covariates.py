"""
testing of covariate multipliers.
"""
import pytest

from cascade.model import covariates
from cascade.model.grids import PriorGrid, AgeTimeGrid
from cascade.model.priors import Gaussian
from cascade.model.rates import Smooth
from cascade.core.context import ModelContext


def test_assign_covariates_to_iota():
    """Demonstration of how covariate assignment works"""
    income = covariates.Covariate("income")
    income.reference = 1000
    income.max_difference = None

    model = ModelContext()

    at_grid = AgeTimeGrid.uniform(
        age_start=0, age_end=120, age_step=5,
        time_start=1990, time_end=2018, time_step=1)
    value_priors = PriorGrid(at_grid)
    value_priors[:, :].prior = Gaussian(0, 1.0)
    at_priors = PriorGrid(at_grid)
    at_priors[:, :].prior = Gaussian(0, 0.1)

    income_time_tight = covariates.CovariateMultiplier(
        income, Smooth(value_priors, at_priors, at_priors)
    )
    # There isn't much to test about the lists of covariate multipliers.
    # They are lists and would permit, for instance, adding the same one twice.
    model.rates.iota.covariate_multipliers.append(income_time_tight)
    model.integrand_covariate_multipliers["remission"].value_covariate_multipliers.append(income_time_tight)
    model.integrand_covariate_multipliers["prevalence"].std_covariate_multipliers.append(income_time_tight)


@pytest.mark.parametrize("cov,ref,diff", [
    ("income", 1000, None),
    ("income", 0, None),
    ("income", 1000, 500),
    ("income", 1000, 2000),
    ("income", "1000", 2000),
    ("income", 1000, "2000"),
    ("income", 1000, 1000),
    ("income", -1000, 1000),
    ("income", 1000, 0.00001),
    ("income", 0, 0.000001),
    ("income", 0, 0),
    ("income", 1000, 0),
])
def test_create_covariate(cov, ref, diff):
    """Tests boundary cases for definition of covariate column"""
    income = covariates.Covariate(cov)
    income.reference = ref
    income.max_difference = diff
    covariates.Covariate(cov, ref, diff)
# What happens downstream if you don't set the reference on a covariate column?
# When does that get checked?


@pytest.mark.parametrize("name,ref,diff", [
    (33, 1000, 500),
    ("income", "howdy", 500),
    ("income", 1000, "doody"),
    ("income", None, 2000),
    ("income", 1000, -1000),
    ("income", -1000, -1000),
    ("", 1000, None),
])
def test_fail_to_create_covariate(name, ref, diff):
    """Failure conditions to reject construction of covariate column"""
    with pytest.raises((ValueError, TypeError)):
        income = covariates.Covariate(name)
        income.reference = ref
        income.max_difference = diff


def test_covariate_column_equality():
    """Equality between two versions of covariate columns"""
    income1 = covariates.Covariate("income", 1000)
    income2 = covariates.Covariate("income", 1000)
    income3 = covariates.Covariate("income", -1000, 500)
    assert income1 == income2
    assert income2 != income3
