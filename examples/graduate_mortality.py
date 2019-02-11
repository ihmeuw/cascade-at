"""
This takes a set of observed mortality rates,
and fits them to find the total mortality rate.
It sets up a Dismod-AT model that has only nonzero omega and treats
the mortality rate observations as mtother.
"""
import itertools as it
import logging

import db_queries
import numpy as np
import pandas as pd
from numpy import nan

from cascade.model import (
    Model, Session, DismodGroups, Var, SmoothGrid,
    Uniform, Gaussian
)


def get_asdr_data(location_and_children, gbd_round_id):
    """Gets the age-specific death rate from IHME databases.
    This is ${}_nm_x$, the mortality rate.
    """
    demo_dict = db_queries.get_demographics(gbd_team="epi", gbd_round_id=gbd_round_id)
    age_group_ids = demo_dict["age_group_id"]
    sex_ids = demo_dict["sex_id"]

    asdr = db_queries.get_envelope(
        location_id=location_and_children,
        year_id=-1,
        gbd_round_id=gbd_round_id,
        age_group_id=age_group_ids,
        sex_id=sex_ids,
        with_hiv=True,
        rates=True,
    ).drop(columns=["run_id"])

    asdr = asdr[asdr["mean"].notnull()]

    return asdr


def bounds_to_stdev(df):
    """Given an upper and lower bound, calculate a standard deviation."""
    with_std = df.assign(std=(df.upper - df.lower) / (2 * 1.96))
    return with_std.drop(columns=["upper", "lower"])


def asdr_by_sex(asdr, ages, sex_id):
    """Incoming age-specific death rate has ``age_id`` and upper and lower
    bounds. This translates those into age-ranges, time-ranges, and standard
    deviations."""
    without_weight = ages.drop(columns=["age_group_weight_value"])
    as_up_low = without_weight.rename({"age_group_years_start": "age_lower", "age_group_years_end": "age_upper"},
                                      axis="columns")
    with_ages = asdr.merge(as_up_low, on="age_group_id", how="left")
    with_upper = with_ages.assign(time_upper=with_ages.year_id + 1)
    with_times = with_upper.rename(columns=dict(year_id="time_lower"))
    with_std = bounds_to_stdev(with_times)
    rest = with_std.assign(
        integrand="mtother",
        location=location_id,
        hold_out=0,
        density="gaussian",
        eta=nan,
        nu=nan,
    )
    trimmed = rest.drop(columns=["age_group_id", "location_id"])
    return trimmed[trimmed.sex_id == sex_id].drop(columns=["sex_id"])


# %%
logging.basicConfig(level=logging.DEBUG)
location_id = 101
gbd_round_id = 5
asdr = get_asdr_data([location_id], gbd_round_id)
ages = db_queries.get_age_metadata(age_group_set_id=12, gbd_round_id=gbd_round_id)
assert not (set(asdr.age_group_id.unique()) - set(ages.age_group_id.values))

# %%
mtother = asdr_by_sex(asdr, ages, 1)


# %%
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


# %%
# Locations, ages, and times on which the model will be based.
locations = pd.DataFrame(dict(
    location_id=[location_id],
    parent_id=[nan],
    name=["country"],
))
ages = np.sort(np.unique(np.concatenate([mtother.age_lower, mtother.age_upper])))
times = np.sort(np.unique(np.concatenate([mtother.time_lower, mtother.time_upper])))

# %%
# Fitting mortality rate depends on prevalence because mortality is an
# integrand. The prevalence is what determines the weight to calculate the
# integrand. This is circular. Here, we use mortality rate as a guess
# at mortality, use that to predict an approximate prevalence. Then
# we will pass that approximate prevalence into the fit as a weight.

initial_mtother_guess = rectangular_data_to_var(mtother)

susceptible_places = pd.DataFrame(dict(
    integrand="susceptible",
    location=location_id,
    age_lower=np.tile(ages, len(times)),
    age_upper=np.tile(ages, len(times)),
    time_lower=np.repeat(times, len(ages)),
    time_upper=np.repeat(times, len(ages)),
))

# Let step size be smallest age group size. No harm because predict is fast.
# If you forget to change this for the fit, memory will explode.
ode_step_size_for_predict = min(np.diff(ages))
print(f"ode step size for predict {ode_step_size_for_predict}")
full_guess = DismodGroups()
full_guess.rate["omega"] = initial_mtother_guess

session = Session(locations=locations, parent_location=location_id, filename="graduate.db")
session.set_option(ode_step_size=ode_step_size_for_predict)
predicted, not_predicted = session.predict(full_guess, susceptible_places, location_id)
# XXX predicted_df is coming back with a sample_index of None. Fix that.
print(f"Smallest predicted susceptible {predicted['mean'].min()}")

# %%
# We are constructing a weight from the prediction, so it can't be zero.
# How about 1 in 10,000 alive as the minimum?
with_min = predicted.assign(floor_value=1e-4)
the_two = with_min[["floor_value", "mean"]]
floored_susceptible = predicted.assign(mean=the_two.max(axis=1))
weight = rectangular_data_to_var(floored_susceptible)
# This assigns two of the four weights (constant, with_condition). Those missing
# will be assigned constant values.
weights = dict(total=weight, susceptible=weight)

# %%

# These are parameters on the fit.
value_stdev = 1e-1  # stdev on the value parameter, if we use Gaussian
dage_ratio = 50  # stdev on dage is dage_ratio * value_stdev
dtime_ratio = 50  # stdev on dtime is dtime_ratio * value_stdev
absolute_min_std = 1e-4  # Change data std to be at least this large.
relative_min_std = 0.005  # and at least this fraction of the mean.
fit_step_size = 5
extra_ages = [0.019, 0.077, 1]

# We will want to set the weight for "total".
# The weight used for mtother is listed in cascade.dismod.constants.INTEGRAND_TO_WEIGHT
model = Model(nonzero_rates=["omega"], parent_location=location_id, child_location=[],
              covariates=None, weights=weights)
omega_grid = SmoothGrid(ages=ages, times=times)
omega_grid.value[:, :] = Uniform(lower=0, upper=1.5, mean=0.01)
# omega_grid.value[:, :] = Gaussian(lower=0, upper=1.5, mean=0.01, standard_deviation=value_stdev)
# XXX This for-loop sets the mean as the initial guess because the fit command
# needs the initial var and scale var to be on the same age-time grid, and
# this set is not. The session could switch it to the other age-time grid.
for age, time in omega_grid.age_time():
    omega_grid.value[age, time] = omega_grid.value[age, time].assign(mean=initial_mtother_guess(age, time))

omega_grid.dage[:, :] = Gaussian(mean=0.0, standard_deviation=dage_ratio * value_stdev)
omega_grid.dtime[:, :] = Gaussian(mean=0.0, standard_deviation=dage_ratio * value_stdev)
model.rate["omega"] = omega_grid

fit_option = dict(random_seed=0,
                  ode_step_size=fit_step_size,
                  age_avg_split=" ".join(str(ea) for ea in extra_ages),
                  # quasi_fixed="true",
                  derivative_test_fixed="none",
                  max_num_iter_fixed=100,
                  print_level_fixed=5,
                  tolerance_fixed=1e-8,
                  )
print(f"age avg split str {fit_option['age_avg_split']}")
session.set_option(**fit_option)

# The data has some small stdevs. Let's give those a smallest value.
smallest_stdev = mtother.assign(rtol=relative_min_std * mtother["mean"], atol=absolute_min_std)
the_three = smallest_stdev[["rtol", "atol", "std"]]
less_stringent = mtother.assign(std=the_three.max(axis=1))

# XXX Make session docs reflect exact data columns.
print("starting fit")
fit_result = session.fit(model, less_stringent)
print(f"finishing fit {fit_result.success}")

# %%
# How much different is the fit_result.fit_residual from the predicted value
# using finer steps? It won't be very different. Under 10%.
session.set_option(ode_step_size=ode_step_size_for_predict)
# XXX make the session docs reflect exact columns and remove columns from
# dataframe b/c that allows us to pass in data as avgints.
avgint = mtother[["integrand", "location", "age_lower", "age_upper", "time_lower", "time_upper"]]
mt_fit, mt_not_fit = session.predict(fit_result.fit, avgint, location_id)

# XXX not handling the not_predicted case.

# %%
simulate_cnt = 5
session.set_option(**fit_option)
simulate_result = session.simulate(model, less_stringent, fit_result.fit, simulate_cnt)

draws = list()
for draw_idx in range(simulate_result.count):
    sim_model, sim_data = simulate_result.simulation(draw_idx)
    # let's start a new session because the simulation results are associated
    # with a session and running a new fit will delete them.
    sim_session = Session(locations=locations, parent_location=location_id, filename="simulate.db")
    sim_session.set_option(**fit_option)
    sim_fit_result = sim_session.fit(sim_model, sim_data)
    draws.append(sim_fit_result.fit)
    # XXX make the Session close or be a contextmanager.
    del sim_session

# %%
# Given draws from a solution, let's set parameters on priors of a new model.
# Don't assume that the model has the same ages and times, or that it has
# the same distributions. Let the draws, as continuous functions, prime the
# next priors.

sub_model = Model(
    nonzero_rates=["omega"], parent_location=location_id, child_location=[],
    covariates=None, weights=weights)
omega_grid = SmoothGrid(ages=ages, times=times)
omega_grid.value[:, :] = Gaussian(lower=0, upper=1.5, mean=0.01, standard_deviation=0.5)
omega_grid.dage[:, :] = Gaussian(mean=0.0, standard_deviation=dage_ratio * value_stdev)
omega_grid.dtime[:, :] = Gaussian(mean=0.0, standard_deviation=dage_ratio * value_stdev)
sub_model.rate["omega"] = omega_grid

for group_name, group in sub_model.items():
    for key, prior_grid in group.items():
        # Gather data from incoming draws.
        draw_data = np.zeros((len(draws), len(prior_grid.ages), len(prior_grid.times)))
        for didx in range(len(draws)):
            one_draw = draws[didx][group_name][key]
            for aidx, age in enumerate(prior_grid.ages):
                for tidx, time in enumerate(prior_grid.times):
                    draw_data[didx, aidx, tidx] = one_draw(age, time)
        draw_data = draw_data.transpose([1, 2, 0])
        draw_dage = np.diff(draw_data, n=1, axis=0)
        draw_dtime = np.diff(draw_data, n=1, axis=1)

        for aidx, tidx in it.product(range(len(ages)), range(len(times))):
            age = prior_grid.ages[aidx]
            time = prior_grid.times[tidx]
            distribution = prior_grid.value[age, time]
            prior_grid.value[age, time] = distribution.mle(draw_data[aidx, tidx, :])

        for aidx, tidx in it.product(range(len(ages) - 1), range(len(times))):
            age = prior_grid.ages[aidx]
            time = prior_grid.times[tidx]
            distribution = prior_grid.dage[age, time]
            prior_grid.dage[age, time] = distribution.mle(draw_dage[aidx, tidx, :])

        for aidx, tidx in it.product(range(len(ages)), range(len(times) - 1)):
            age = prior_grid.ages[aidx]
            time = prior_grid.times[tidx]
            distribution = prior_grid.dtime[age, time]
            prior_grid.dtime[age, time] = distribution.mle(draw_dtime[aidx, tidx, :])

# %%

sub_session = Session(locations=locations, parent_location=location_id, filename="subsession.db")
sub_session.set_option(**fit_option)
sub_fit = sub_session.fit(sub_model, less_stringent)
print(f"Sub fit converged {sub_fit.success}")
