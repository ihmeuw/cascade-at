import pandas as pd

from cascade.testing_utilities import make_execution_context
from cascade.core.context import ModelContext

from cascade.model.integrands import integrand_grids_from_gbd


def test_integrand_grids_from_gbd(mocker):
    get_age_groups = mocker.patch("cascade.model.integrands.get_age_groups")
    get_years = mocker.patch("cascade.model.integrands.get_years")

    get_age_groups.return_value = pd.DataFrame(
        [[0, 1], [1, 4], [4, 82]], columns=["age_group_years_start", "age_group_years_end"]
    )
    get_years.return_value = [1990, 1995, 2000]

    mc = ModelContext()
    ec = make_execution_context()

    integrand_grids_from_gbd(mc, ec)

    for integrand in mc.outputs.integrands:
        assert set(integrand.age_ranges) == {(0, 1), (1, 4), (4, 82)}
        assert set(integrand.time_ranges) == {(1990, 1990), (1995, 1995), (2000, 2000)}
