from types import SimpleNamespace
from cascade.input_data.configuration.construct_mortality import (
    get_raw_csmr, normalize_csmr
)
import pytest
from cascade.executor.execution_context import make_execution_context
from cascade.core.db import age_spans


@pytest.mark.parametrize("location_id,cause_id", [
    (137, 468),
    (101, 529),
])
def test_live_csmr(ihme, location_id, cause_id):
    ec = make_execution_context()
    data_access = SimpleNamespace()
    data_access.tier = 2
    data_access.add_csmr_cause = cause_id
    data_access.cod_version = 90
    data_access.gbd_round_id = 6
    data_access.location_set_version_id = 429
    ages = age_spans.get_age_spans()
    raw = get_raw_csmr(ec, data_access, location_id, ages)
    normed = normalize_csmr(raw, [1, 2])
    assert not set(normed.sex_id.unique().tolist()) - {1, 2}
    expected_cols = {
        "location_id", "sex_id", "mean", "age_lower", "age_upper",
        "time_lower", "time_upper", "lower", "upper", "measure", "hold_out",
    }
    assert set(normed.columns.tolist()) == expected_cols
