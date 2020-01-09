import pytest


def test_complete_covariate_ages(covariate_data):
    df = covariate_data.complete_covariate_ages(cov_df=covariate_data.raw)
    assert all([c in df.age_group_id.unique()
                for c in covariate_data.demographics.age_group_id])


@pytest.fixture(scope='module')
def df_for_dismod(covariate_data, population, dag):
    return covariate_data.configure_for_dismod(
        pop_df=population.raw,
        loc_df=dag.df
    )


def test_square_covariates(df_for_dismod, Demographics):
    assert len(df_for_dismod) == (
        3 *  # for the 3 sexes, always
        len(Demographics.location_id) *
        len(Demographics.age_group_id)
    )


@pytest.mark.parametrize("column,value", [
    ("location_id", 70.),
    ("year_id", 1990.),
    ("age_group_id", 2.),
    ("sex_id", 2.),
    ("mean_value", 0.96),
    ("age_lower", 0.),
    ("age_upper", 0.01917808)
])
def test_all_columns(df_for_dismod, column, value):
    df = df_for_dismod.loc[
        (df_for_dismod.age_group_id == 2) & (df_for_dismod.sex_id == 2)
    ].copy()
    assert df[column].iloc[0] == value
