from cascade.executor.epiviz_runner import add_mortality_data, add_omega_constraint
from cascade.testing_utilities import make_execution_context
from cascade.core.context import ModelContext

import pandas as pd
import pytest


@pytest.mark.parametrize("sexes", [1, 2])
def test_add_mortality_data(ihme, sexes):
    ec = make_execution_context(model_version_id=265976, gbd_round_id=5)
    mc = ModelContext()
    mc.input_data.observations = pd.DataFrame(
        columns=["time_upper", "time_lower", "age_upper", "age_lower", "measure", "hold_out"]
    )
    mc.policies["estimate_emr_from_prevalence"] = False

    add_mortality_data(mc, ec, sexes)
    with_mort = mc.input_data.observations
    assert "sex_id" in with_mort.columns
    assert not with_mort.query("sex_id == @sexes").empty
    other_sex = {1: 2, 2: 1}.get(sexes)  # noqa: F841
    assert with_mort.query("sex_id == @other_sex").empty


@pytest.mark.parametrize("sexes", [1, 2])
def test_add_omega_constraint(ihme, sexes):
    ec = make_execution_context(model_version_id=265976, gbd_round_id=5)
    mc = ModelContext()
    mc.input_data.observations = pd.DataFrame(
        {"time_lower": [1970.0],
         "time_upper": [1971.0],
         "age_lower": [0.0],
         "age_upper": [1.0],
         "measure": "prevalence",
         "hold_out": [0],
         })
    add_omega_constraint(mc, ec, sexes)
    assert not mc.input_data.observations.empty
    assert mc.rates.omega.parent_smooth is not None
