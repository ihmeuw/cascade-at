import pytest

import pandas as pd

from cascade.input_data.db.demographics import age_ranges_to_groups


def test_age_ranges_to_groups(mocker):
    fake_get_age_groups = mocker.patch("cascade.input_data.db.demographics.get_age_groups")
    fake_get_age_groups.return_value = pd.DataFrame(
        {"age_group_id": [1, 2, 3, 4], "age_group_years_start": [0, 1, 25, 90], "age_group_years_end": [1, 2, 30, 100]}
    )

    df = pd.DataFrame({"age_lower": [0, 1, 90], "age_upper": [1, 2, 100], "integrand": ["remission"] * 3})

    new_df = age_ranges_to_groups(None, df)

    assert new_df.age_group_id.tolist() == [1, 2, 4]
    assert set(new_df.columns) == {"age_group_id", "integrand"}

    new_df = age_ranges_to_groups(None, df, True)
    assert set(new_df.columns) == {"age_group_id", "integrand", "age_lower", "age_upper"}


def test_age_ranges_to_groups__birth(mocker):
    fake_get_age_groups = mocker.patch("cascade.input_data.db.demographics.get_age_groups")
    fake_get_age_groups.return_value = pd.DataFrame()

    df = pd.DataFrame({"age_lower": [0], "age_upper": [0], "integrand": ["prevalence"]})

    new_df = age_ranges_to_groups(None, df)
    assert new_df.age_group_id.tolist() == [164]

    df["integrand"] = "remission"
    with pytest.raises(ValueError):
        age_ranges_to_groups(None, df)
