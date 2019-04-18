import pandas as pd
from numpy import nan
from cascade.input_data.configuration.construct_study import (
    add_study_covariate_to_observations
)


def test_no_stray_columns_when_excluded():
    observations = pd.DataFrame(dict(
        integrand=["prevalence", "remission", "remission", "prevalence", "prevalence"],
        location=89,
        density="gaussian",
        eta=nan,
        nu=nan,
        age_lower=[13.0, 55, 55, 55, 65],
        age_upper=[18.0, 85, 85, 64, 74],
        time_lower=[1995.0, 1990, 1990, 1990, 1990],
        time_upper=[1995.0, 2000, 2000, 1993, 1993],
        mean=[0.105, 0.116, 0.048, 0.042, 0.103],
        std=[0.018, 0.069, 0.028, 0.009, 0.0145],
        sex_id=[3, 1, 2, 1, 1],
        name=["88", "96", "97", "183", "184"],
        seq=[88.0, 96, 97, 183, 184],
        hold_out=0,
    ))
    study_covariates = pd.DataFrame(dict(
        bundle_id=4310,
        seq=[88, 96, 97, 183, 183],
        study_covariate_id=[215, 233, 233, 246, 1192],
    ))
    id_to_name = {0: "s_sex", 1604: "s_one"}
    obs = add_study_covariate_to_observations(observations, study_covariates, id_to_name)
    assert not any([c for c in obs.columns if isinstance(c, int)])
    assert set(obs.columns) == set(id_to_name.values()) | set(observations.columns)


def test_includes_requested_column():
    observations = pd.DataFrame(dict(
        integrand=["prevalence", "remission", "remission", "prevalence", "prevalence"],
        location=89,
        density="gaussian",
        eta=nan,
        nu=nan,
        age_lower=[13.0, 55, 55, 55, 65],
        age_upper=[18.0, 85, 85, 64, 74],
        time_lower=[1995.0, 1990, 1990, 1990, 1990],
        time_upper=[1995.0, 2000, 2000, 1993, 1993],
        mean=[0.105, 0.116, 0.048, 0.042, 0.103],
        std=[0.018, 0.069, 0.028, 0.009, 0.0145],
        sex_id=[3, 1, 2, 1, 1],
        name=["88", "96", "97", "183", "184"],
        seq=[88.0, 96, 97, 183, 184],
        hold_out=0,
    ))
    study_covariates = pd.DataFrame(dict(
        bundle_id=4310,
        seq=[88, 96, 97, 183, 183],
        study_covariate_id=[215, 233, 233, 246, 1192],
    ))
    id_to_name = {0: "s_sex", 1604: "s_one", 233: "borlaug"}
    obs = add_study_covariate_to_observations(observations, study_covariates, id_to_name)
    assert not any([c for c in obs.columns if isinstance(c, int)])
    assert set(obs.columns) == set(id_to_name.values()) | set(observations.columns)
