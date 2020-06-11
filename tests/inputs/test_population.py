import pytest


@pytest.fixture
def df_for_dismod(population):
    return population.configure_for_dismod()


@pytest.mark.parametrize("column,value", [
    ("age_lower", 0.),
    ("age_upper", 0.01917808),
    ("age_group_id", 2),
    ("location_id", 70),
    ("year_id", 1990),
    ("population", 3000.)
])
def test_all_columns(df_for_dismod, column, value):
    # This is just testing the very first
    # row of the population dataset.
    assert df_for_dismod[column].iloc[0] == value
