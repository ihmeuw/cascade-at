import numpy as np
import pandas as pd

from cascade.dismod.constants import IntegrandEnum
from cascade.model.integrands import make_average_integrand_cases_from_gbd


def test_make_average_integrand_cases_from_gbd(mocker):
    age_groups = pd.DataFrame(
        [[0, 1], [1, 4], [4, 82]], columns=["age_group_years_start", "age_group_years_end"]
    )
    years = [1990, 1995, 2000]

    parent_location_id = 180

    average_integrand_cases = make_average_integrand_cases_from_gbd(
        age_groups, years, [1, 2], parent_location_id)

    assert np.all(average_integrand_cases.location == 180)
    for (age_lower, age_upper) in {(0, 1), (1, 4), (4, 82)}:
        for (time_lower, time_upper) in {(1990, 1990), (1995, 1995), (2000, 2000)}:
            for integrand in IntegrandEnum:
                assert len(
                    average_integrand_cases.query(
                        "age_lower == @age_lower and age_upper == @age_upper "
                        "and time_lower == @time_lower and time_upper == @time_upper "
                        "and integrand == @integrand.name"
                    )
                    == 1
                )


def test_make_average_integrand_cases_from_gbd__with_birth_prevalence(mocker):
    age = pd.DataFrame(
        [[0, 1], [1, 4], [4, 82]], columns=["age_group_years_start", "age_group_years_end"]
    )
    years = [1990, 1995, 2000]

    parent_location_id = 180

    average_integrand_cases = make_average_integrand_cases_from_gbd(
        age, years, [1, 2], parent_location_id, include_birth_prevalence=True)

    birth_rows = average_integrand_cases.query("age_lower == 0 and age_upper == 0")
    assert all(birth_rows.integrand == "prevalence")
    assert len(birth_rows) == 3 * 2

    assert np.all(average_integrand_cases.location == 180)
    for (age_lower, age_upper) in {(0, 1), (1, 4), (4, 82)}:
        for (time_lower, time_upper) in {(1990, 1990), (1995, 1995), (2000, 2000)}:
            for integrand in IntegrandEnum:
                assert len(
                    average_integrand_cases.query(
                        "age_lower == @age_lower and age_upper == @age_upper "
                        "and time_lower == @time_lower and time_upper == @time_upper "
                        "and integrand == @integrand.name"
                    )
                    == 1
                )
