"""
This takes a set of observed mortality rates,
and fits them to find the total mortality rate.
It sets up a Dismod-AT model that has only nonzero omega and treats
the mortality rate observations as mtother.
"""
import itertools as it
import logging
from timeit import default_timer as timer

import db_queries
import numpy as np
import pandas as pd
from numpy import nan

from cascade.model import (
    Model, Session, DismodGroups, Var, SmoothGrid,
    Uniform, Gaussian
)
from cascade.input_data.db.asdr import asdr_as_fit_input

LOGGER = logging.getLogger(__name__)


def construct_weights(initial_mtother_guess, locations, ages, times, location_id, step_size):
    """A weight is a function of age and time that is the population of a state.
    For incidence, that state is susceptible. For excess mortality, that
    state is with-condition. It is the state individuals leave. For total
    mortality, it's susceptible plus with-condition.

    This function makes a rough estimate of the population size in order to use
    it as a weight on total mortality.
    """
    susceptible_places = pd.DataFrame(dict(
        integrand="susceptible",
        location=location_id,
        age_lower=np.tile(ages, len(times)),
        age_upper=np.tile(ages, len(times)),
        time_lower=np.repeat(times, len(ages)),
        time_upper=np.repeat(times, len(ages)),
    ))
    full_guess = DismodGroups()
    full_guess.rate["omega"] = initial_mtother_guess
    session = Session(locations=locations, parent_location=location_id, filename="graduate.db")
    session.set_option(ode_step_size=step_size)
    begin = timer()
    predicted, not_predicted = session.predict(full_guess, susceptible_places, location_id)
    LOGGER.info(f"predict {timer() - begin}")
    # XXX predicted_df is coming back with a sample_index of None. Fix that.
    print(f"Smallest predicted susceptible {predicted['mean'].min()}")
    # We are constructing a weight from the prediction, so it can't be zero.
    # How about 1 in 10,000 alive as the minimum?
    with_min = predicted.assign(floor_value=1e-4)
    the_two = with_min[["floor_value", "mean"]]
    floored_susceptible = predicted.assign(mean=the_two.max(axis=1))
    weight = rectangular_data_to_var(floored_susceptible)
    # This assigns two of the four weights (constant, with_condition). Those missing
    # will be assigned constant values.
    return dict(total=weight, susceptible=weight)


def rectangular_data_to_var(gridded_data):
    """Using this very regular data, where every age and time is present,
    construct an initial guess as a Var object. Very regular means that there
    is a complete set of ages-cross-times."""
    initial_ages = np.sort(np.unique(0.5 * (gridded_data.age_lower + gridded_data.age_upper)))
    initial_times = np.sort(np.unique(0.5 * (gridded_data.time_lower + gridded_data.time_upper)))

    guess = Var(ages=initial_ages, times=initial_times)
    for age, time in guess.age_time():
        found = gridded_data.query(
            "(age_lower <= @age) & (@age <= age_upper) & (time_lower <= @time) & (@time <= time_upper)")
        assert len(found) == 1, f"found {found}"
        guess[age, time] = float(found.iloc[0]["mean"])
    return guess


def estimate_mortality_hazard(mtother, initial_mtother_guess, weights, params):
    # We will want to set the weight for "total".
    # The weight used for mtother is listed in cascade.dismod.constants.INTEGRAND_TO_WEIGHT
    ages = params["ages"]
    times = params["times"]
    model = Model(nonzero_rates=["omega"], parent_location=params["location_id"], child_location=[],
                  covariates=None, weights=weights)
    omega_grid = SmoothGrid(ages=ages, times=times)
    omega_grid.value[:, :] = Uniform(lower=0, upper=1.5, mean=0.01)
    # omega_grid.value[:, :] = Gaussian(lower=0, upper=1.5, mean=0.01, standard_deviation=value_stdev)
    # XXX This for-loop sets the mean as the initial guess because the fit command
    # needs the initial var and scale var to be on the same age-time grid, and
    # this set is not. The session could switch it to the other age-time grid.
    for age, time in omega_grid.age_time():
        omega_grid.value[age, time] = omega_grid.value[age, time].assign(mean=initial_mtother_guess(age, time))

    omega_grid.dage[:, :] = Gaussian(mean=0.0, standard_deviation=params["dage_ratio"] * params["value_stdev"])
    omega_grid.dtime[:, :] = Gaussian(mean=0.0, standard_deviation=params["dtime_ratio"] * params["value_stdev"])
    model.rate["omega"] = omega_grid

    # The data has some small stdevs. Let's give those a smallest value.
    smallest_stdev = mtother.assign(rtol=params["relative_min_std"] * mtother["mean"], atol=params["absolute_min_std"])
    the_three = smallest_stdev[["rtol", "atol", "std"]]
    less_stringent = mtother.assign(std=the_three.max(axis=1))

    # XXX Make session docs reflect exact data columns.
    session = Session(locations=params["locations"], parent_location=params["location_id"], filename="graduate.db")
    session.set_option(**params["fit_option"])
    begin = timer()
    fit_result = session.fit(model, less_stringent)
    LOGGER.info(f"fit {timer() - begin} Success {fit_result.success}")
    max_fit = fit_result.fit

    # How much different is the fit_result.fit_residual from the predicted value
    # using finer steps? It won't be very different. Under 10%.
    session.set_option(ode_step_size=params["ode_step_size_for_predict"])
    # XXX make the session docs reflect exact columns and remove columns from
    # dataframe b/c that allows us to pass in data as avgints.
    avgint = mtother[["integrand", "location", "age_lower", "age_upper", "time_lower", "time_upper"]]
    mt_fit, mt_not_fit = session.predict(max_fit, avgint, params["location_id"])

    # XXX not handling the not_predicted case.
    draws = make_draws(model, less_stringent, max_fit, params["simulate_cnt"], params)

    return max_fit, draws


def make_draws(model, less_stringent, max_fit, count, params):
    session = Session(locations=params["locations"], parent_location=params["location_id"], filename="simulate.db")
    session.set_option(**params["fit_option"])
    simulate_result = session.simulate(model, less_stringent, max_fit, count)

    draws = list()
    for draw_idx in range(simulate_result.count):
        sim_model, sim_data = simulate_result.simulation(draw_idx)
        # let's start a new session because the simulation results are associated
        # with a session and running a new fit will delete them.
        fit_file = f"simulate{draw_idx}.db"
        sim_session = Session(locations=locations, parent_location=location_id, filename=fit_file)
        sim_session.set_option(**params["fit_option"])
        begin = timer()
        sim_fit_result = sim_session.fit(sim_model, sim_data)
        LOGGER.info(f"fit {timer() - begin} success {sim_fit_result.success}")
        if sim_fit_result.success:
            draws.append(sim_fit_result.fit)
            print(f"sim fit {draw_idx} success")
        else:
            print(f"sim fit {draw_idx} not successful in {fit_file}.")
        # XXX make the Session close or be a contextmanager.
        del sim_session
    return draws


def estimate_mortality_with_draws(mtother, draws, weights, params):
    # Given draws from a solution, let's set parameters on priors of a new model.
    # Don't assume that the model has the same ages and times, or that it has
    # the same distributions. Let the draws, as continuous functions, prime the
    # next priors.
    ages = params["ages"]
    times = params["times"]

    sub_model = Model(
        nonzero_rates=["omega"], parent_location=params["location_id"], child_location=[],
        covariates=None, weights=weights)
    omega_grid = SmoothGrid(ages=ages, times=times)
    omega_grid.value[:, :] = Gaussian(lower=0, upper=1.5, mean=0.01, standard_deviation=0.5)
    omega_grid.dage[:, :] = Gaussian(mean=0.0, standard_deviation=params["dage_ratio"] * params["value_stdev"])
    omega_grid.dtime[:, :] = Gaussian(mean=0.0, standard_deviation=params["dage_ratio"] * params["value_stdev"])
    sub_model.rate["omega"] = omega_grid

    set_priors_from_draws(sub_model, draws)

    # The data has some small stdevs. Let's give those a smallest value.
    smallest_stdev = mtother.assign(rtol=params["relative_min_std"] * mtother["mean"], atol=params["absolute_min_std"])
    the_three = smallest_stdev[["rtol", "atol", "std"]]
    less_stringent = mtother.assign(std=the_three.max(axis=1))

    sub_session = Session(locations=locations, parent_location=location_id, filename="subsession.db")
    sub_session.set_option(**params["fit_option"])
    begin = timer()
    sub_fit = sub_session.fit(sub_model, less_stringent)
    LOGGER.info(f"fit {timer() - begin} success {sub_fit.success}")

    draws = make_draws(sub_model, less_stringent, sub_fit.fit, params["simulate_cnt"], params)

    return sub_fit, draws


def set_priors_from_draws(model, draws):
    """Sets priors from posteriors of the *same model*."""
    if len(draws) == 0:
        return

    for group_name, group in model.items():
        if group_name not in draws[0]:
            continue

        for key, prior_grid in group.items():
            if key not in draws[0][group_name]:
                continue

            ages = prior_grid.ages
            times = prior_grid.times
            draw_value, draw_dage, draw_dtime = gather_draws_for_grid(draws, group_name, key, ages, times)

            estimate_grid_parameters(prior_grid.value, draw_value, ages, times)
            estimate_grid_parameters(prior_grid.dage, draw_dage, ages[:-1], times)
            estimate_grid_parameters(prior_grid.dtime, draw_dtime, ages, times[:-1])


def set_priors_from_parent_draws(model, draws):
    """Sets priors from posteriors of the *parent model*."""
    assert len(draws) > 0

    for group_name, group in model.items():
        if group_name not in draws[0] or group_name == "random_effect":
            continue

        for key, prior_grid in group.items():
            if key not in draws[0][group_name]:
                continue

            ages = prior_grid.ages
            times = prior_grid.times
            if group_name == "rate" and (key, model.location_id) in draws[0]["random_effect"]:
                draw_value, draw_dage, draw_dtime = gather_draws_for_child_grid(
                    draws, group_name, key, ages, times, location_id)
                LOGGER.debug(f"Child prior found for {group_name} {key}")
            elif group_name != "rate":
                draw_value, draw_dage, draw_dtime = gather_draws_for_grid(draws, group_name, key, ages, times)
                LOGGER.debug(f"Prior found for {group_name} {key}")
            else:
                LOGGER.debug(f"No prior found for {group_name} {key}")
                continue

            estimate_grid_parameters(prior_grid.value, draw_value, ages, times)
            estimate_grid_parameters(prior_grid.dage, draw_dage, ages[:-1], times)
            estimate_grid_parameters(prior_grid.dtime, draw_dtime, ages, times[:-1])


def gather_draws_for_grid(draws, group_name, key, ages, times):
    # Gather data from incoming draws into an array of (draw, age, time)
    draw_data = np.zeros((len(draws), len(ages), len(times)))
    for didx in range(len(draws)):
        one_draw = draws[didx][group_name][key]
        for aidx, age in enumerate(ages):
            for tidx, time in enumerate(times):
                draw_data[didx, aidx, tidx] = one_draw(age, time)

    draw_data = draw_data.transpose([1, 2, 0])
    draw_dage = np.diff(draw_data, n=1, axis=0)
    draw_dtime = np.diff(draw_data, n=1, axis=1)
    return draw_data, draw_dage, draw_dtime


def gather_draws_for_child_grid(draws, group_name, key, ages, times, location_id):
    # Gather data from incoming draws into an array of (draw, age, time)
    draw_data = np.zeros((len(draws), len(ages), len(times)))
    for didx in range(len(draws)):
        underlying = draws[didx][group_name][key]
        random_effect = draws[didx]["random_effect"][(key, location_id)]
        for aidx, age in enumerate(ages):
            for tidx, time in enumerate(times):
                draw_data[didx, aidx, tidx] = underlying(age, time) * np.exp(random_effect(age, time))

    draw_data = draw_data.transpose([1, 2, 0])
    draw_dage = np.diff(draw_data, n=1, axis=0)
    draw_dtime = np.diff(draw_data, n=1, axis=1)
    return draw_data, draw_dage, draw_dtime


def estimate_grid_parameters(grid_priors, draws, ages, times):
    for aidx, tidx in it.product(range(len(ages)), range(len(times))):
        age = ages[aidx]
        time = times[tidx]
        grid_priors[age, time] = grid_priors[age, time].mle(draws[aidx, tidx, :])


# %%
logging.root.setLevel(logging.INFO)
location_id = 101  # Canadia
sex_id = 1
gbd_round_id = 5
decomp_step = "step1"
age_group_set_id = 12

ages_df = db_queries.get_age_metadata(age_group_set_id=age_group_set_id, gbd_round_id=gbd_round_id)
# This comes in yearly from 1950 to 2018
mtother = asdr_as_fit_input(location_id, sex_id, gbd_round_id, decomp_step, ages_df, with_hiv=True)
# Reduce years by factor because it's slow with too much data.
# Maybe smarter to work with a dense set of years, so limit to 1990-2000?
mtother = mtother[(mtother.time_lower % 10) < 0.1]


# %%
# Locations, ages, and times on which the model will be based.
locations = pd.DataFrame(dict(
    location_id=[location_id],
    parent_id=[nan],
    name=["country"],
))
ages = np.sort(np.unique(np.concatenate([mtother.age_lower, mtother.age_upper])))
times = np.sort(np.unique(mtother.time_lower))
times = np.concatenate([times, times[-1:] + 1])
LOGGER.info(f"age cnt {len(ages)} time cnt {times}")
# Let step size be smallest age group size. No harm because predict is fast.
# If you forget to change this for the fit, memory will explode.
ode_step_size_for_predict = min(np.diff(ages))
print(f"ode step size for predict {ode_step_size_for_predict}")

# Fitting mortality rate depends on prevalence because mortality is an
# integrand. The prevalence is what determines the weight to calculate the
# integrand. This is circular. Here, we use mortality rate as a guess
# at mortality, use that to predict an approximate prevalence. Then
# we will pass that approximate prevalence into the fit as a weight.

initial_mtother_guess = rectangular_data_to_var(mtother)

weights = construct_weights(initial_mtother_guess, locations, ages, times, location_id, ode_step_size_for_predict)

# %%

# These are parameters on the fit.
params = dict(
    ages=ages,
    times=times,
    locations=locations,
    location_id=location_id,
    value_stdev=1e-1,  # stdev on the value parameter, if we use Gaussian
    dage_ratio=50,  # stdev on dage is dage_ratio * value_stdev
    dtime_ratio=50,  # stdev on dtime is dtime_ratio * value_stdev
    absolute_min_std=1e-4,  # Change data std to be at least this large.
    relative_min_std=0.005,  # and at least this fraction of the mean.
    fit_step_size=5,
    simulate_cnt=5,
    extra_ages=[0.019, 0.077, .15, .5, 1, 2],
    ode_step_size_for_predict=ode_step_size_for_predict,
)
params["fit_option"] = dict(
    random_seed=0,
    ode_step_size=params["fit_step_size"],
    age_avg_split=" ".join(str(ea) for ea in params["extra_ages"]),
    # quasi_fixed="true",
    derivative_test_fixed="none",
    max_num_iter_fixed=100,
    print_level_fixed=5,
    tolerance_fixed=1e-8,
)
print(f"age avg split str {params['fit_option']['age_avg_split']}")

# %%

max_fit, draws = estimate_mortality_hazard(mtother, initial_mtother_guess, weights, params)
second_fit, second_draws = estimate_mortality_with_draws(mtother, draws, weights, params)

# %%
third_fit, third_draws = estimate_mortality_with_draws(mtother, second_draws, weights, params)
