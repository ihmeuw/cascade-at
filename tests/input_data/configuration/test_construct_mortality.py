from types import SimpleNamespace
from cascade.input_data.configuration.construct_mortality import (
    normalize_csmr
)
from cascade.input_data.db.csmr import get_csmr_location
import pytest
from cascade.executor.execution_context import make_execution_context
from cascade.core.db import age_spans
from cascade.input_data.configuration.construct_country import (
    convert_gbd_ids_to_dismod_values
)


@pytest.mark.parametrize("location_id,cause_id", [
    (137, 468),
    (101, 529),
])
def test_live_csmr(ihme, location_id, cause_id):
    ec = make_execution_context()
    data_access = SimpleNamespace()
    data_access.tier = 2
    data_access.add_csmr_cause = cause_id
    cod_version = 90
    gbd_round_id = 6
    decomp_step = "step1"
    data_access.location_set_version_id = 429
    ages = age_spans.get_age_spans()
    raw_csmr = get_csmr_location(ec, location_id, cause_id, cod_version, gbd_round_id, decomp_step)
    raw = convert_gbd_ids_to_dismod_values(raw_csmr, ages)
    normed = normalize_csmr(raw, [1, 2])
    assert not set(normed.sex_id.unique().tolist()) - {1, 2}
    expected_cols = {
        "location_id", "sex_id", "mean", "age_lower", "age_upper",
        "time_lower", "time_upper", "lower", "upper", "measure", "hold_out",
    }
    assert set(normed.columns.tolist()) == expected_cols
