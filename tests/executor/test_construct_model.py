import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from numpy.random import RandomState

from cascade.executor.cascade_plan import CascadePlan
from cascade.executor.construct_model import construct_model
from cascade.executor.create_settings import (
    create_local_settings, create_settings, SettingsChoices, make_locations
)
from cascade.executor.dismodel_main import parse_arguments
from cascade.executor.session_options import make_options
from cascade.input_data.db.locations import location_hierarchy_to_dataframe
from cascade.model.session import Session
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
    session.set_option(**make_options(local_settings.settings))
    session.setup_model_for_fit(model)


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


def change_setting(settings, name, value):
    members = name.split(".")
    obj = None
    for m in members[:-1]:
        if obj is None:
            obj = getattr(settings, m)
        else:
            obj = getattr(obj, m)
    setattr(obj, members[-1], value)


def test_construct_model_fair(dismod, tmp_path):
    lose_file = True
    filename = tmp_path / "z.db" if lose_file else "model_fair.db"
    rng = RandomState(424324)
    for i in range(10):
        construct_model_fair(filename, rng.get_state())


def test_same_settings(dismod, tmp_path, base_settings, reference_db):
    filename = tmp_path / "single_settings.db"
    local_settings, locations = make_local_settings(base_settings)
    make_a_db(local_settings, locations, filename)

    compare = CompareDatabases("reference_comparison.db", filename)
    assert not compare.table_diffs()
    assert not compare.different_tables()


@pytest.mark.parametrize("setstr,val,opt", [
    ("settings.policies.meas_std_effect", "add_std_scale_all", "meas_std_effect"),
    ("settings.model.zero_sum_random", "iota omega chi".split(), "zero_sum_random"),
    ("settings.model.ode_step_size", 0.5, "ode_step_size"),
    ("settings.model.additional_ode_steps", [2.7, 3.4], "age_avg_split"),
    ("settings.model.random_seed", 2342987, "random_seed"),
    ("settings.model.quasi_fixed", 1, "quasi_fixed"),
    ("settings.model.bound_frac_fixed", 1e-4, "bound_frac_fixed"),
    ("settings.policies.limited_memory_max_history_fixed", 50, "limited_memory_max_history_fixed"),
    ("settings.model.bound_random", 0.2, "bound_random"),
    ("settings.derivative_test.fixed", "first-order", "derivative_test_fixed"),
    ("settings.derivative_test.random", "first-order", "derivative_test_random"),
    ("settings.max_num_iter.fixed", 10, "max_num_iter_fixed"),
    ("settings.max_num_iter.random", 10, "max_num_iter_random"),
    ("settings.print_level.fixed", 4, "print_level_fixed"),
    ("settings.print_level.random", 4, "print_level_random"),
    ("settings.accept_after_max_steps.fixed", 6, "accept_after_max_steps_fixed"),
    ("settings.accept_after_max_steps.random", 6, "accept_after_max_steps_random"),
    ("settings.tolerance.fixed", 1.23, "tolerance_fixed"),
    ("settings.tolerance.random", 1.23, "tolerance_random"),
])
def test_option_settings(dismod, tmp_path, base_settings, reference_db, setstr, val, opt):
    filename = tmp_path / "single_settings.db"
    local_settings, locations = make_local_settings(base_settings)
    change_setting(local_settings, setstr, val)
    make_a_db(local_settings, locations, filename)

    compare = CompareDatabases("reference_comparison.db", filename)
    assert not compare.table_diffs()
    assert compare.different_tables() & {"option"}
    print(compare.record_differences("option"))
    assert compare.diff_contains("option", opt)
