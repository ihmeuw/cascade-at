from pathlib import Path
import pickle
from types import SimpleNamespace

from numpy.random import RandomState
import pytest

from cascade.executor.cascade_plan import CascadePlan
from cascade.executor.dismodel_main import parse_arguments
from cascade.executor.construct_model import construct_model
from cascade.executor.create_settings import (
    create_local_settings, create_settings, SettingsChoices, make_locations
)
from cascade.model.session import Session
from cascade.input_data.db.locations import location_hierarchy_to_dataframe
from cascade.testing_utilities.compare_dismod_db import CompareDatabases


@pytest.fixture
def base_settings():
    return """
    iota = True
    rho = False
    omega = True
    chi = True
    pini = False

    emr = 0
    constrain_omega = 1
    iota.at_specific = 0
    iota.min = 0.0001
    iota.age_cnt = 2
    iota.time_cnt = 2
    omega.at_specific = 1
    omega.min = 0.0001
    omega.age_cnt = 0
    omega.time_cnt = 0
    chi.at_specific = 0
    chi.min = 0.0001
    chi.age_cnt = 1
    chi.time_cnt = 2
    drill_start = 2
    drill_end = 14
    re.iota = all
    re.omega = all
    re.chi = all
    study.1604 = True
    study.1604.at_specific = 0
    study.1604.age_cnt = 1
    study.1604.time_cnt = 1
    study.1604.covtype = meas_std
    study.2453 = False
    study.6497 = False
    country.1604 = True
    country.1604.at_specific = 0
    country.1604.age_cnt = 1
    country.1604.time_cnt = 1
    country.1604.covtype = rate_value
    country.1604.rate = iota
    country.2453 = True
    country.2453.at_specific = 0
    country.2453.age_cnt = 1
    country.2453.time_cnt = 1
    country.2453.covtype = meas_std
    country.6497 = False
    job_idx = 0
    """


@pytest.fixture
def reference_db(base_settings):
    local_settings, locations = make_local_settings(base_settings)
    make_a_db(local_settings, locations, "reference_comparison.db")


def test_construct_model_fair(dismod, tmp_path):
    lose_file = True
    filename = tmp_path / "z.db" if lose_file else "model_fair.db"
    rng = RandomState(424324)
    for i in range(10):
        construct_model_fair(filename, rng.get_state())


def construct_model_fair(filename, rng_state):
    rng = RandomState()
    rng.set_state(rng_state)
    local_settings, locations = create_local_settings(rng)
    data = SimpleNamespace()
    data.locations = locations
    model = construct_model(data, local_settings)
    assert len(model.rate.keys()) > 0
    session = Session(location_hierarchy_to_dataframe(locations),
                      parent_location=1, filename=filename)
    try:
        session.setup_model_for_fit(model)
    except AssertionError:
        pickle.dump(rng_state, Path("fail_state.pkl").open("wb"))
        raise


def test_single_settings(dismod, tmp_path, base_settings, reference_db):
    filename = tmp_path / "single_settings.db"
    local_settings, locations = make_local_settings(base_settings)
    make_a_db(local_settings, locations, filename)

    compare = CompareDatabases("reference_comparison.db", filename)
    print(compare.table_diffs)


def make_local_settings(given_settings):
    choices = SettingsChoices(settings=given_settings)
    args = parse_arguments(["z.db"])
    depth = 4
    locations = make_locations(depth)
    settings = create_settings(choices, locations)
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    j = list(c.cascade_jobs)[1:]
    job_choice = choices.choice(list(range(len(j))), name="job_idx")
    job_kind, job_args = c.cascade_job(j[job_choice])
    assert job_kind == "estimate_location"
    return job_args, locations


def make_a_db(local_settings, locations, filename):
    data = SimpleNamespace()
    data.locations = locations
    model = construct_model(data, local_settings)
    session = Session(location_hierarchy_to_dataframe(locations),
                      parent_location=1, filename=filename)
    session.setup_model_for_fit(model)
