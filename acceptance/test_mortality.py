from subprocess import run
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from db_queries import get_envelope, get_outputs

from cascade.executor.epiviz_runner import add_settings_to_execution_context
from cascade.input_data.db.configuration import load_settings
from cascade.dismod.constants import IntegrandEnum
from cascade.input_data.db.demographics import age_groups_to_ranges
from cascade.dismod.db.wrapper import DismodFile, _get_engine
from cascade.testing_utilities import make_execution_context
from cascade.core.db import cursor


def _prepare_output_data_for_test(execution_context, clear_bundle=False):
    mvid = execution_context.parameters.model_version_id

    # Clear out all the tier 3 tables so we are looking at fresh data
    table_names = ["t3_model_version_asdr", "t3_model_version_csmr", "t3_model_version_study_covariate"]
    if clear_bundle:
        # rebuilding the t3 bundle can be expensive so only clear it if we need to
        table_names.append("t3_model_version_dismod")
    for table_name in [
        "t3_model_version_asdr",
        "t3_model_version_csmr",
        "t3_model_version_dismod",
        "t3_model_version_study_covariate",
    ]:
        clear_table = f"delete from {table_name} where model_version_id = {mvid}"
        with cursor(execution_context) as c:
            c.execute(clear_table)

    with TemporaryDirectory() as d:
        db_path = Path(d) / "test.db"
        run(f"dmcascade {db_path} --mvid {mvid} --db-only".split(), check=True)
        dm = DismodFile(_get_engine(db_path))
        return dm.data


def test_asdr_data():
    mvid = 266156
    execution_context = make_execution_context()
    settings = load_settings(execution_context, mvid=mvid)
    add_settings_to_execution_context(execution_context, settings)

    data = _prepare_output_data_for_test(execution_context)
    data = data.loc[data.integrand_id == IntegrandEnum.mtall.value]

    expected = get_envelope(
        location_id=execution_context.parameters.parent_location_id,
        year_id="all",
        gbd_round_id=execution_context.parameters.gbd_round_id,
        age_group_id="all",
        sex_id="all",
        with_hiv=True,
        rates=True,
    )

    expected = age_groups_to_ranges(execution_context, expected)

    # The actual years used depend on the input bundle. I'm going to assume that part works
    years = data.time_lower.unique()  # noqa: F841
    expected = expected.query("year_id in @years")

    expected = expected.query("sex_id == @settings.model.drill_sex")

    expected_mean = expected.sort_values(["age_lower", "year_id"]).reset_index(drop=True)["mean"]

    actual_mean = data.sort_values(["age_lower", "time_lower"]).reset_index(drop=True)["meas_value"]

    assert np.allclose(expected_mean, actual_mean)


def test_csmr_data():
    mvid = 266156
    execution_context = make_execution_context()
    settings = load_settings(execution_context, mvid=mvid)
    add_settings_to_execution_context(execution_context, settings)

    data = _prepare_output_data_for_test(execution_context)
    data = data.loc[data.integrand_id == IntegrandEnum.mtspecific.value]

    expected = get_outputs(
        topic="cause",
        cause_id=execution_context.parameters.add_csmr_cause,
        location_id=execution_context.parameters.parent_location_id,
        metric_id=3,
        year_id="all",
        age_group_id="most_detailed",
        measure_id=1,
        sex_id="all",
        gbd_round_id=execution_context.parameters.gbd_round_id,
        version="latest",
    )
    expected = expected[expected["val"].notnull()]

    expected = age_groups_to_ranges(execution_context, expected)

    expected = expected.query("sex_id == @settings.model.drill_sex")

    expected_mean = expected.sort_values(["age_lower", "year_id"]).reset_index(drop=True)["val"]

    actual_mean = data.sort_values(["age_lower", "time_lower"]).reset_index(drop=True)["meas_value"]

    assert np.allclose(expected_mean, actual_mean)
