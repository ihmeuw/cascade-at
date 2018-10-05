import numpy as np
import pandas as pd

from cascade.testing_utilities import make_execution_context
from cascade.dismod.db.metadata import IntegrandEnum

from cascade.model.integrands import make_average_integrand_cases_from_gbd


def test_make_average_integrand_cases_from_gbd(mocker):
    get_age_groups = mocker.patch("cascade.model.integrands.get_age_groups")
    get_years = mocker.patch("cascade.model.integrands.get_years")

    get_age_groups.return_value = pd.DataFrame(
        [[0, 1], [1, 4], [4, 82]], columns=["age_group_years_start", "age_group_years_end"]
    )
    get_years.return_value = [1990, 1995, 2000]

    ec = make_execution_context(location_id=180)

    average_integrand_cases = make_average_integrand_cases_from_gbd(ec)

    assert np.all(average_integrand_cases.node_id == 180)
    for (age_lower, age_upper) in {(0, 1), (1, 4), (4, 82)}:
        for (time_lower, time_upper) in {(1990, 1990), (1995, 1995), (2000, 2000)}:
            for integrand in IntegrandEnum:
                assert len(
                    average_integrand_cases.query(
                        "age_lower == @age_lower and age_upper == @age_upper "
                        "and time_lower == @time_lower and time_upper == @time_upper "
                        "and integrand_name == @integrand.name"
                    )
                    == 1
                )
