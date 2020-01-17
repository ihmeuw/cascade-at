"""
testing of covariate multipliers.
"""
import pytest
from numpy import nan

from cascade_at.model import covariate


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
    income = covariate.Covariate(cov)
    income.reference = ref
    income.max_difference = diff
    covariate.Covariate(cov, ref, diff)
# What happens downstream if you don't set the reference on a covariate column?
# When does that get checked?


def test_set_nan():
    c = covariate.Covariate("chewing_gum", 0, nan)
    assert c.max_difference is None
    assert c.reference == 0


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
        income = covariate.Covariate(name)
        income.reference = ref
        income.max_difference = diff


def test_covariate_column_equality():
    """Equality between two versions of covariate columns"""
    income1 = covariate.Covariate("income", 1000)
    income2 = covariate.Covariate("income", 1000)
    income3 = covariate.Covariate("income", -1000, 500)
    assert income1 == income2
    assert income2 != income3
