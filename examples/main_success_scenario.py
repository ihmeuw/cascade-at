"""
Predict data and then fit it.
This application is a way to see whether the set of steps,
that we consider the main work of the wrapper, will work for
a range of parameters. It's set up to work through different
sets of parameters and, if something breaks, to give you a good
way to rerun the failing set with modifications.

Each run writes two kinds of files, a db file, which it will
delete if all goes well, and a json file with timings for
a run of the fit.
"""
import json
import logging
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from copy import copy
from getpass import getuser
from itertools import product, combinations
from math import nan, inf
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.random import RandomState

from cascade.core import getLoggers
from cascade.dismod import DismodATException
from cascade.dismod.constants import INTEGRAND_COHORT_COST
from cascade.dismod.process_behavior import get_fit_output
from cascade.model import (
    Model, SmoothGrid, Covariate, DismodGroups, Var,
    ObjectWrapper, Gaussian, Uniform
)

CODELOG, MATHLOG = getLoggers(__name__)
DB_FILE_LOCATION = Path("/ihme/scratch/users") / getuser() / "db"
TIMING_FILE_LOCATION = Path("timing")  # A relative path to current working


def reasonable_grid_from_var(var, age_time, group_name):
    """
    This constructs the smooth_grid for which that var is the single true fit.

    Args:
        var (Var): A single var grid.
        age_time (List[float],List[float]): Tuple of ages and times
            on which to put the prior distributions.
        group_name (str): "rate", "random_effect", "alpha", "beta", "gamma"
            The var is one of the smooth grids in this group. We need to
            know the group in order to know what kinds of distributions
            are allowed and whether values can be less than zero.

    Returns:
        SmoothGrid: A single smooth grid with priors that should
        solve for the given Var.
    """
    if age_time is None:
        age_time = (var.ages, var.times)
    smallest_rate = 1e-9
    min_meas_std = 1e-4
    difference_stdev = 0.2  # Can be large.
    smooth_grid = SmoothGrid(*age_time)
    if group_name == "rate":
        for age, time in smooth_grid.age_time():
            meas_value = max(var(age, time), smallest_rate)
            # Meas value should ramp from zero up.
            meas_std = 0.1 * meas_value + min_meas_std
            smooth_grid.value[age, time] = Gaussian(
                meas_value, meas_std, lower=smallest_rate, upper=meas_value * 5)
    elif group_name == "random_effect":
        for age, time in smooth_grid.age_time():
            meas_value = var(age, time)
            # meas_value should be near zero, so add some stdev
            meas_std = 0.1 * meas_value + 0.1
            # No upper or lower bounds on random effect distributions.
            smooth_grid.value[age, time] = Gaussian(meas_value, meas_std)
    elif group_name in {"alpha", "beta", "gamma"}:
        for age, time in smooth_grid.age_time():
            meas_value = var(age, time)
            meas_std = 0.1 * meas_value + 0.1
            # These can have upper and lower bounds.
            smooth_grid.value[age, time] = Gaussian(
                meas_value, meas_std, lower=-1, upper=1)
    else:
        raise RuntimeError(f"Unknown group_name {group_name}")

    # Set all grid points so that the edges are set.
    smooth_grid.dage[:, :] = Uniform(-inf, inf, 0)
    smooth_grid.dtime[:, :] = Uniform(-inf, inf, 0)
    # Then calculate actual dage and dtime.
    ages = smooth_grid.ages
    times = smooth_grid.times
    mid_time = times[0]
    for age_start, age_finish in zip(ages[:-1], ages[1:]):
        dage = var(age_finish, mid_time) - var(age_start, mid_time)
        smooth_grid.dage[age_start, :] = Gaussian(dage, difference_stdev)
        # Rough check that the sign is right.
        assert dage >= -0.1, f"dage {var.ages}"
    mid_age = ages[0]
    for time_start, time_finish in zip(times[:-1], times[1:]):
        dtime = var(mid_age, time_finish) - var(mid_age, time_start)
        assert dtime >= -0.1
        smooth_grid.dtime[:, time_start] = Gaussian(dtime, difference_stdev)

    return smooth_grid


def pretty_print_groups(groups, indent=""):
    for group_name, group in groups.items():
        CODELOG.debug(f"{indent}{group_name}")
        for key, item in group.items():
            CODELOG.debug(f"{indent}  {key}: {item}")


def model_from_var(var_groups, parent_location, age_time=None,
                   multiple_random_effects=False, covariates=None):
    """
    Given values across all rates, construct a model with loose priors
    in order to be able to predict from those rates.

    Args:
        var_groups (DismodGroups[Var]): Values on grids.
        parent_location (int): A parent location, because that isn't
            in the keys.
        age_time (Tuple[List[float],List[float]]): Ages and times to use
            for the priors that this creates. If None, use the var ages
            and times.
        multiple_random_effects (bool): Create a separate smooth grid for
            each random effect in the Var. The default is to create a
            single smooth grid for all random effects.
        covariates (List[Covariate]): List of covariate names.
    Returns:
        Model: with Uniform distributions everywhere and no mulstd.
    """
    child_locations = list(sorted({k[1] for k in var_groups.random_effect.keys()}))
    nonzero_rates = list(var_groups.rate.keys())
    model = Model(
        nonzero_rates,
        parent_location,
        child_locations,
        covariates=covariates,
    )

    # Cycle through rate, random_effect, alpha, beta, gamma.
    for group_name, group in var_groups.items():
        is_random_effect = group_name == "random_effect"
        skip_re_children = is_random_effect and not multiple_random_effects
        # Within each of the five, cycle through rate surfaces.
        for key, item_var in group.items():
            if skip_re_children:
                # Make one smooth grid for all children.
                assign_key = (key[0], None)
            else:
                assign_key = key

            if assign_key not in model[group_name]:
                model[group_name][assign_key] = reasonable_grid_from_var(
                    item_var, age_time, group_name)

    pretty_print_groups(var_groups)
    pretty_print_groups(model)
    return model


TOPOLOGY = dict(
    no_remission=["iota", "chi", "omega"],
    single_measure_a=["omega"],
    single_measure_b=["iota"],
    no_death=["iota", "rho"],
    illness_death=["iota", "chi", "rho", "omega"],
)
"""There are 16 ways to set up rates on Susceptible-Condition-Removed, and these
are the ones that makes some sense. There are two topologies that never
converge, so they are removed and considered a modeling question:
having birth prevalence but no incidence (rho, omega, chi), and
having no remission at all (omega, chi).
"""


TOPOLOGY_FORBIDDEN_INTEGRANDS = dict(
    no_remission=["remission"],
    single_measure_a=[
        "Sincidence", "remission", "mtexcess", "withC", "prevalence",
        "Tincidence", "mtspecific", "mtstandard", "relrisk"
    ],
    single_measure_b=[
        "remission", "mtexcess", "withC", "prevalence",
        "Tincidence", "mtspecific", "mtstandard", "relrisk", "mtother"
    ],
    born=["Sincidence", "Tincidence", "remission"],
    no_death=["mtexcess", "mtother", "mtspecific", "mtall", "mtstandard", "relrisk"],
    born_remission=["Sincidence", "Tincidence"],
    illness_death=[],
)
"""These are kinds of data you CANNOT have for each topology."""


CHOICES = dict(
    n_children=[8, 4, 2, 16, 32],
    topology_choice=list(TOPOLOGY.keys()),
    age_cnt=[8, 16, 32],
    time_cnt=[2, 4, 8, 16, 32],
    covariate_cnt=[4, 1, 2, 0, 8, 16],
    fit_kind=["both", "fixed", "random"],
    percent_alpha_covariate=[1, 0.6, 0.3, 0],
    cohort_cost=["no", "both", "yes"],
    data_at_extent=[True, False],
    zero_sum_random=[True, False],
    ode_step_size=[10, 20, 5, 2, 1],
    quasi_fixed=["false", "true"],
    # Walk iterations at the same time, and always walk them.
    data_cnt=[100, 500, 200, 1000, 50],
)
"""The first choice in each list will be the default.
All combinations are excursions from that default.
"""


def all_choices(level_cnt=2):
    """Constructs a list of experiments from the choices given above.
    The level is the order of the fractional-factorial experiment.
    So 2 means we look at all combinations of interactions
    of two of the parameters at a time.
    """
    categories = CHOICES.keys()
    total = [dict()]  # Start with default settings.
    for levels in range(1, level_cnt + 1):
        for multiple_cats in combinations(categories, levels):
            multiple_values = [CHOICES[c] for c in multiple_cats]
            for combo in product(*multiple_values):
                total.append(dict(zip(multiple_cats, combo)))
    with_iterations = list()
    default_choices = {key: values[0] for (key, values) in CHOICES.items()}
    # For every combination of parameters, it's important to know the difference
    # between setup time and iteration time, so that we can choose the
    # number of iterations, so we always go through three iterations counts.
    for single_run in total:
        for iter_cnt in [1, 20, 40]:
            new_run = copy(default_choices)
            new_run.update(single_run)
            new_run["max_num_iter_fixed"] = iter_cnt
            new_run["max_num_iter_random"] = iter_cnt
            with_iterations.append(new_run)

    return with_iterations


def fit_sim(settings, rng):
    """user_fit_sim.py from Dismod-AT done with file movement."""
    n_children = settings.n_children  # You can change the number of children.
    topology_choice = settings.topology_choice
    age_cnt = settings.age_cnt
    time_cnt = settings.time_cnt
    covariate_cnt = settings.covariate_cnt
    topology = TOPOLOGY[topology_choice]
    if settings.cohort_cost == "yes":
        data_integrands = [k for (k, v) in INTEGRAND_COHORT_COST.items() if v]
    elif settings.cohort_cost == "no":
        data_integrands = [k for (k, v) in INTEGRAND_COHORT_COST.items() if not v]
    elif settings.cohort_cost == "both":
        data_integrands = list(INTEGRAND_COHORT_COST.keys())
    else:
        raise RuntimeError(f"Unrecognized cohort cost {settings.cohort_cost}")

    # If there's no remission, you can't have remission data.
    forbidden_integrands = TOPOLOGY_FORBIDDEN_INTEGRANDS[topology_choice]
    data_integrands = list(set(data_integrands) - set(forbidden_integrands))

    parent_location = 1
    child_locations = [parent_location + 1 + add_child
                       for add_child in range(n_children)]
    locations = pd.DataFrame(dict(
        name=["Universe"] + [f"child_{cidx}" for cidx in range(n_children)],
        parent_id=[nan] + n_children * [parent_location],
        location_id=[parent_location] + child_locations,
    ))

    base_year = 2000
    covariates = [
        Covariate(f"sdi{cov_create_idx}", reference=0.1)
        for cov_create_idx in range(covariate_cnt)
    ]

    # The model sits on one age and time and has only incidence rate, iota.
    truth_var = DismodGroups()
    if "iota" in topology:
        iota = Var([0, 50], [base_year])
        iota[0, :] = 0
        iota[50, :] = 0.01
        truth_var.rate["iota"] = iota
    if "rho" in topology:
        rho = Var([50], [base_year])
        rho[:, :] = 0.005
        truth_var.rate["rho"] = rho
    if "omega" in topology:
        omega = Var([0, 1, 10, 100], [base_year])
        omega[0, base_year] = 0.08
        omega[1, base_year] = 0.01
        omega[10, base_year] = 0.0001
        omega[100, base_year] = 0.05
        truth_var.rate["omega"] = omega
    if "chi" in topology:
        chi = Var([0, 100], [base_year])
        chi[0, base_year] = 0.0
        chi[100, base_year] = 0.05
        truth_var.rate["chi"] = chi

    for random_effect_rate, re_child in product(topology, child_locations):
        random_effect = Var([50], [base_year])
        random_effect[:, :] = 0.05
        truth_var.random_effect[(random_effect_rate, re_child)] = random_effect

    for make_cov_idx, make_cov_var in enumerate(covariates):
        cov_var = Var([50], base_year)
        cov_var[:, :] = make_cov_var.reference
        if make_cov_idx < settings.percent_alpha_covariate * len(covariates):
            apply_to = rng.choice(topology)
            truth_var.alpha[(make_cov_var.name, apply_to)] = cov_var
        else:
            apply_to = rng.choice(data_integrands)
            truth_var.beta[(make_cov_var.name, apply_to)] = cov_var

    model = model_from_var(
        truth_var, parent_location, covariates=covariates)

    if Path("/ihme").exists():
        db_dir = DB_FILE_LOCATION
    else:
        db_dir = Path("timing_db_files")
    db_dir.mkdir(exist_ok=True, parents=True)
    db_file = db_dir / f"{uuid4()}.db"
    if db_file.exists():
        db_file.unlink()
    settings.db_file = str(db_file)
    dismod_objects = ObjectWrapper(db_file)
    dismod_objects.locations = locations
    dismod_objects.parent_location_id = parent_location
    dismod_objects.model = model
    dismod_objects.age_extents = [0, 100]
    dismod_objects.time_extents = [1990, 2010]

    dismod_objects.run_dismod("init")
    dismod_objects.truth_var = truth_var

    data_ages = np.linspace(0, 100, 21)
    data_times = np.linspace(1990, 2010, 5)
    covariate_values = {
        make_cov_val.name: 0.1
        for make_cov_val in covariates
    }
    data_predictions = pd.DataFrame([dict(
        integrand=rng.choice(data_integrands),
        location=rng.choice(child_locations),
        age=rng.choice(data_ages),
        time=rng.choice(data_times),
        **covariate_values,
    )
        for _i in range(settings.data_cnt)
    ])
    if settings.data_at_extent:
        data_predictions = data_predictions.assign(
            age_lower=(data_predictions.age - 1).clip(0, 100),
            age_upper=(data_predictions.age + 1).clip(0, 100),
            time_lower=(data_predictions.time - 1).clip(1990, 2010),
            time_upper=(data_predictions.time + 1).clip(1990, 2010),
        )
        data_predictions = data_predictions.drop(columns=["age", "time"])

    dismod_objects.avgint = data_predictions
    dismod_objects.run_dismod(["predict", "truth_var"])
    predicted, not_predicted = dismod_objects.predict
    print(predicted.columns)

    data = predicted.drop(columns=["sample_index"])
    data = data.assign(
        density="gaussian",
        std=data["mean"] * 0.1 + 0.001,
        eta=1e-5,
    )
    dismod_objects.data = data

    # Now make a new model with different ages and times.
    ages = np.linspace(0, 100, age_cnt)
    times = np.linspace(1990, 2010, time_cnt)
    model = model_from_var(
        truth_var,
        parent_location,
        age_time=(ages, times),
        covariates=covariates
    )
    dismod_objects.model = model
    if settings.zero_sum_random:
        zero_sum_list = " ".join(topology)
    else:
        zero_sum_list = nan
    option = dict(random_seed=0,
                  ode_step_size=settings.ode_step_size,
                  zero_sum_random=zero_sum_list,
                  quasi_fixed=settings.quasi_fixed,
                  derivative_test_fixed="first-order",
                  max_num_iter_fixed=100,
                  print_level_fixed=5,
                  tolerance_fixed=1e-8,
                  derivative_test_random="second-order",
                  max_num_iter_random=100,
                  tolerance_random=1e-8,
                  print_level_random=0,
                  )
    dismod_objects.set_option(**option)
    dismod_objects.run_dismod("init")
    requires_prerun = settings.fit_kind != "fixed"
    if requires_prerun:
        fit_kind, settings.fit_kind = settings.fit_kind, "fixed"
        run_and_record_fit(dismod_objects, settings)
        settings.fit_kind = fit_kind
        dismod_objects.start_var = dismod_objects.fit_var
        dismod_objects.scale_var = dismod_objects.fit_var
    run_and_record_fit(dismod_objects, settings)


def run_and_record_fit(dismod_objects, settings):
    dismod_at_choice = f"fit {settings.fit_kind}"
    CODELOG.info(dismod_at_choice)
    stdout, stderr, metrics = dismod_objects.run_dismod(dismod_at_choice)
    exit_kind, exit_string, iteration_cnt = get_fit_output(stdout)
    print(f"{exit_string} with {iteration_cnt} iterations")
    metrics["ipopt iterations"] = iteration_cnt
    # The originally-intended settings can augment this dictionary.
    metrics.update(settings.__dict__)
    print(metrics)
    timing_dir = TIMING_FILE_LOCATION
    timing_dir.mkdir(exist_ok=True, parents=True)
    timing_path = timing_dir / f"{uuid4()}.json"
    with timing_path.open("w") as timing_out:
        json.dump(
            metrics,
            timing_out,
            default=json_translate,
            indent=2,
        )
        timing_out.write(os.linesep)


def json_translate(o):
    """This exists because we try to write an np.int64 to JSON."""
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def make_parser():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0,
                        help="rerun with same seed")
    parser.add_argument("--choice", type=int, default=-1,
                        help="Rerun a particular choice_idx")
    # These are all in order to modify one value in a run.
    # The point is that, if you see a failing run, you can, by hand
    # reduce the number of data points or time points until it
    # fails _faster_, and then use that set of parameters and db file for
    # debugging.
    for param, param_values in CHOICES.items():
        param_default = param_values[0]
        arg = param.replace("_", "-")
        parser.add_argument(f"--{arg}", type=type(param_default),
                            required=False)
    parser.add_argument("--choices", action="store_true",
                        help="Show how many choices there are")
    return parser


def configure_sim(args):
    if args.seed != 0:
        rng_seed = args.seed
    else:
        rng = RandomState()
        # Choose a seed so that we know how to reseed to get the same run.
        rng_seed = rng.randint(2342434)
    print(f"seed is {rng_seed}")
    rng = RandomState(rng_seed)
    every_choice = all_choices()
    task_id = os.environ.get("SGE_TASK_ID", None)
    if args.choice >= 0:
        choice_idx = args.choice
    elif task_id is not None:
        choice_idx = (int(task_id) - 1) % len(every_choice)
    else:
        choice_idx = rng.randint(len(every_choice))
    CODELOG.info(f"Using choice {choice_idx} of {len(every_choice)}")
    chosen = every_choice[choice_idx]
    for underscore_arg in chosen.keys():
        if hasattr(args, underscore_arg) and getattr(args, underscore_arg):
            chosen[underscore_arg] = getattr(args, underscore_arg)
    # Add these so they show up in the json metrics output.
    chosen["rng_seed"] = rng_seed
    chosen["choice_idx"] = choice_idx
    chosen["task_id"] = task_id
    settings = SimpleNamespace(**chosen)
    CODELOG.info(settings)
    return settings, rng


def entry():
    args = make_parser().parse_args()
    level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level=level)
    if args.choices:
        print(len(all_choices()))
        exit(0)
    try:
        sim_settings, sim_rng = configure_sim(args)
        fit_sim(sim_settings, sim_rng)
    except DismodATException as dat_exc:
        print(dat_exc)
        exit(1)


if __name__ == "__main__":
    entry()
