from cascade.executor.epiviz_runner import add_mortality_data, add_omega_constraint
from cascade.testing_utilities import make_execution_context
from cascade.core.context import ModelContext
import cascade.input_data.db.mortality
from cascade.model.priors import Constant
from cascade.input_data.db.locations import get_descendents

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("sexes", [1, 2])
def test_add_mortality_data(ihme, sexes):
    ec = make_execution_context(model_version_id=265976, gbd_round_id=5, location_id=6)
    mc = ModelContext()
    mc.policies = dict(estimate_emr_from_prevalence=0, use_weighted_age_group_midpoints=0)
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
    ec = make_execution_context(model_version_id=265976, gbd_round_id=5, location_id=6)
    mc = ModelContext()
    mc.policies = dict(estimate_emr_from_prevalence=0, use_weighted_age_group_midpoints=0)
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


def test_omega_constraint_as_effect(ihme, monkeypatch):
    """Assert that the omega constraint is an effect"""
    parent_id = 6
    ec = make_execution_context(model_version_id=265976, gbd_round_id=5, location_id=parent_id)
    children = get_descendents(ec, children_only=True)
    assert len(children) > 0
    mc = ModelContext()
    mc.parameters.location_id = parent_id
    mc.policies = dict(estimate_emr_from_prevalence=0, use_weighted_age_group_midpoints=0)
    mc.input_data.observations = pd.DataFrame(
        {"time_lower": [1970.0],
         "time_upper": [1971.0],
         "age_lower": [0.0],
         "age_upper": [1.0],
         "measure": "prevalence",
         "hold_out": [0],
         })

    asdr = cascade.input_data.db.mortality.get_age_standardized_death_rate_data(ec)
    asdr.location_id = parent_id
    child_asdr = [asdr.assign(location_id=child_id) for child_id in children]
    child_asdr.append(asdr)
    same_means = pd.concat(child_asdr, axis=0)
    assert len(same_means) > 0
    assert len(same_means[same_means.location_id == parent_id]) > 0

    def same_omegas(execution_context):
        return same_means

    monkeypatch.setattr(cascade.input_data.db.mortality, "get_age_standardized_death_rate_data", same_omegas)

    add_omega_constraint(mc, ec, 1)
    child_omegas = mc.rates.omega.child_smoothings
    for p in child_omegas[0][1].value_priors.priors:
        assert isinstance(p, Constant) and np.isclose(p.value, 0.0)
