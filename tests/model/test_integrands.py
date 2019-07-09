import pandas as pd

from cascade.dismod.constants import IntegrandEnum
from cascade.model.integrands import make_average_integrand_cases_from_gbd


def test_make_average_integrand_cases_from_gbd():
    age_groups = pd.DataFrame(
        [[0, 1], [1, 4], [4, 82]], columns=["age_group_years_start", "age_group_years_end"]
    )
    years = [1990, 1995, 2000]
    child_location = [2, 3]
    average_integrand_cases = make_average_integrand_cases_from_gbd(
        age_groups, years, [1, 2], child_location)

    assert set(average_integrand_cases.location) == set(child_location)

    for (age_lower, age_upper) in {(0, 1), (1, 4), (4, 82)}:
        for (time_lower, time_upper) in {(1990, 1990), (1995, 1995), (2000, 2000)}:
            for integrand in IntegrandEnum:
                for sex in [1, 2]:
                    for location in child_location:
                        assert len(
                            average_integrand_cases.query(
                                "age_lower == @age_lower and age_upper == @age_upper "
                                "and time_lower == @time_lower and time_upper == @time_upper "
                                "and integrand == @integrand.name "
                                "and location == @location "
                                "and sex_id == @sex"
                            )
                            == 1
                        )
