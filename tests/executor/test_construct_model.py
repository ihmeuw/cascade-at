import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gridengineapp import execution_ordered
from numpy import isclose, inf
from numpy.random import RandomState

from cascade.executor.construct_model import construct_model
from cascade.executor.construct_model import matching_knots
from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.create_settings import (
    create_local_settings, create_settings, SettingsChoices
)
from cascade.executor.dismodel_main import DismodAT
from cascade.executor.estimate_location import modify_input_data
from cascade.executor.execution_context import make_execution_context
from cascade.executor.session_options import make_options
from cascade.input_data.configuration.form import SmoothingPrior
from cascade.input_data.db.locations import location_hierarchy, location_hierarchy_to_dataframe
from cascade.model.session import Session
from cascade.testing_utilities.compare_dismod_db import (
    CompareDatabases, pull_covariate, pull_covariate_multiplier
)
from cascade.testing_utilities.fake_data import retrieve_fake_data


@pytest.fixture
def base_settings():
    """These are default initial values for the random generation of settings.
    Look at create_settings to see what each of these lines sets.
    """
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
    drill_start = 0
    drill_end = -1
    re.iota = all
    re.omega = all
    re.chi = all
    study.0 = False
    study.11 = True
    study.11.at_specific = 0
    study.11.age_cnt = 1
    study.11.time_cnt = 1
    study.11.covtype = rate_value
    study.11.rate = chi
    study.1604 = True
    study.1604.at_specific = 0
    study.1604.age_cnt = 1
    study.1604.time_cnt = 1
    study.1604.covtype = meas_std
    country.156 = True
    country.156.at_specific = 0
    country.156.age_cnt = 1
    country.156.time_cnt = 1
    country.156.covtype = rate_value
    country.156.rate = iota
    country.1998 = True
    country.1998.at_specific = 0
    country.1998.age_cnt = 1
    country.1998.time_cnt = 1
    country.1998.covtype = meas_std
    job_idx = 0
    """


@pytest.fixture
def reference_db(base_settings):
    local_settings, locations = make_local_settings(base_settings)
    make_a_db(local_settings, locations, "reference_comparison.db")


def make_local_settings(given_settings):
    """
    Uses a custom settings generator.
    Skips downloading settings. Instead constructs an application,
    giving it the custom settings. Then gets a graph from that application.
    Then chooses a graph member that does estimation.
    """
    ec = make_execution_context()
    choices = SettingsChoices(settings=given_settings)
    locations = location_hierarchy(gbd_round_id=6, location_set_version_id=429)
    settings = create_settings(choices, locations)

    args = DismodAT.add_arguments().parse_args([])
    app = DismodAT(locations, settings, ec, args)
    task_graph = app.job_graph()

    j = list(execution_ordered(task_graph))[1:]
    estimators = [jest for jest in j
                  if jest.recipe == "estimate_location"]
    job_choice = choices.choice(list(range(len(estimators))), name="job_idx")
    job = app.job(estimators[job_choice])
    return job.local_settings, locations


def make_a_db(local_settings, locations, filename):
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    ec = make_execution_context()
    print("About to retrieve fake data")
    input_data = retrieve_fake_data(ec, local_settings, covariate_data_spec)
    print("About to modify input data")
    modified_data = modify_input_data(input_data, local_settings)
    print("About to construct model")
    model = construct_model(modified_data, local_settings, covariate_multipliers, covariate_data_spec)
    session = Session(location_hierarchy_to_dataframe(locations),
                      parent_location=1, filename=filename)
    session.set_option(**make_options(local_settings.settings, local_settings.model_options))
    session.setup_model_for_fit(model)


def construct_model_fair(ec, filename, rng_state):
    """Thread test from making settings to running the first init."""
    rng = RandomState()
    rng.set_state(rng_state)

    locations = location_hierarchy(gbd_round_id=6, location_set_version_id=429)
    local_settings, locations = create_local_settings(rng, locations=locations)
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    input_data = retrieve_fake_data(ec, local_settings, covariate_data_spec)
    modified_data = modify_input_data(input_data, local_settings)
    model = construct_model(modified_data, local_settings, covariate_multipliers, covariate_data_spec)
    assert len(model.rate.keys()) > 0
    session = Session(location_hierarchy_to_dataframe(locations),
                      parent_location=1, filename=filename)
    try:
        session.setup_model_for_fit(model)
    except AssertionError:
        pickle.dump(rng_state, Path("fail_state.pkl").open("wb"))
        raise

    # These check that covariate data was assigned.
    for dismod_cov_idx in range(len(covariate_data_spec)):
        name, ref, max_diff, values = pull_covariate(filename, dismod_cov_idx)
        if name == "s_one":
            assert all(isclose(v, 1) for v in values)
        elif name == "s_sex":
            assert all(any(isclose(v, x) for x in (-0.5, 0, 0.5)) for v in values)
    with pytest.raises(ValueError):
        # There must not be too many covariates
        pull_covariate(filename, len(covariate_data_spec))

    for mulcov_idx in range(len(covariate_multipliers)):
        pull_covariate_multiplier(filename, mulcov_idx)
    with pytest.raises(ValueError):
        pull_covariate_multiplier(filename, len(covariate_multipliers))


def change_setting_in_local_settings(settings, name, value):
    """
    Assumes the settings object has nested attributes. So if name
    is ``model_options.bound_random`` and value=0.7, then this sets
    ``settings.model_options.bound_random=0.7``.
    """
    members = name.split(".")
    obj = None
    for m in members[:-1]:
        if obj is None:
            obj = getattr(settings, m)
        else:
            obj = getattr(obj, m)
    setattr(obj, members[-1], value)


@pytest.mark.skip("Update for new way to make a db.")
@pytest.mark.parametrize("draw", list(range(10)))
def test_construct_model_fair(ihme, tmp_path, draw):
    lose_file = True
    filename = tmp_path / "z.db" if lose_file else "model_fair.db"
    ec = make_execution_context()
    rng = RandomState(424324 + 979834 * draw)
    construct_model_fair(ec, filename, rng.get_state())


def test_same_settings(ihme, tmp_path, base_settings, reference_db):
    """Shows that if we don't change settings, the tester doesn't see a change."""
    filename = tmp_path / "single_settings.db"
    local_settings, locations = make_local_settings(base_settings)
    make_a_db(local_settings, locations, filename)

    compare = CompareDatabases("reference_comparison.db", filename)
    assert not compare.table_diffs()
    assert not compare.different_tables()


@pytest.mark.parametrize("setstr,val,opt", [
    ("settings.policies.meas_std_effect", "add_std_scale_all", "meas_noise_effect"),
    ("settings.model.zero_sum_random", "iota chi".split(), "zero_sum_random"),
    ("settings.model.ode_step_size", 0.5, "ode_step_size"),
    ("settings.model.additional_ode_steps", [2.7, 3.4], "age_avg_split"),
    ("settings.model.random_seed", 2342987, "random_seed"),
    ("settings.model.quasi_fixed", 1, "quasi_fixed"),
    ("settings.model.bound_frac_fixed", 1e-4, "bound_frac_fixed"),
    ("settings.policies.limited_memory_max_history_fixed", 50, "limited_memory_max_history_fixed"),
    ("model_options.bound_random", 0.23, "bound_random"),
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
def test_option_settings(ihme, tmp_path, base_settings, reference_db, setstr, val, opt):
    """Proves that each of these EpiViz settings makes it all the way to the db file."""
    filename = tmp_path / "single_settings.db"
    local_settings, locations = make_local_settings(base_settings)
    change_setting_in_local_settings(local_settings, setstr, val)
    make_a_db(local_settings, locations, filename)

    compare = CompareDatabases("reference_comparison.db", filename)
    assert not compare.table_diffs()
    print(compare.record_differences("option"))
    assert compare.different_tables() & {"option"}
    assert compare.diff_contains("option", opt)


@pytest.mark.parametrize("kind,al,au,tl,tu,bl,bu,cnt", [
    ("data", 0, 10, 2000, 2040, 1000, inf, 11 * 4),
    ("data", 0, 30, 2000, 2040, 1000, inf, 20 * 4),
    ("data", 0, 10, 1980, 2040, 1000, inf, 11 * 5),
    ("data", 0, 10, 2000, 2040, -inf, inf, 11 * 4),
    ("data", 0, 10, 2000, 2040, 1991, inf, 11 * 4 - 1),
    ("data", 0, 10, 2000, 2040, 1000, 2014, 11 * 4 - 1),
    ("data", 0, 10, 2000, 2040, 1991, 2014, 11 * 4 - 2),
])
def test_matching_knots(kind, al, au, tl, tu, bl, bu, cnt):
    """See if subselection of ages and times by detailed priors works."""
    prior = SmoothingPrior()
    prior_set = dict(
        prior_type=kind,
        age_lower=al,
        age_upper=au,
        time_lower=tl,
        time_upper=tu,
        born_lower=bl,
        born_upper=bu
    )
    for elem, value in prior_set.items():
        setattr(prior, elem, value)
    prior.validate_and_normalize()
    grid = SimpleNamespace()
    grid.ages = np.arange(20)
    grid.times = np.array([1990, 2000, 2005, 2010, 2015])
    age_time = list(matching_knots(grid, prior))
    assert len(age_time) == cnt
    # uniqueness
    assert len({(a, t) for (a, t) in age_time}) == len(age_time)
