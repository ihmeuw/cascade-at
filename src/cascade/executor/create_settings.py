"""
Constructs random models.

rates 5
random effects (child count)
alpha (5*covariates)
beta, gammma (13*covariates)

none or some of children, covariates, integrands

grid at 1 point or >1 point for age or time
priors present or absent on value, dage, dtime
bounds upper/lower can differ between rates, res, and covariates.
"""
from copy import deepcopy
from pprint import pprint

import networkx as nx

from cascade.executor.cascade_plan import CascadePlan
from cascade.executor.dismodel_main import parse_arguments
from cascade.input_data.configuration import SettingsError
from cascade.input_data.db.configuration import json_settings_to_frozen_settings

BASE_CASE = {
    "model": {
        "random_seed": 899736881,
        "default_age_grid": "0 .1 .2 1 5 10 20 30 40 50 60 70 80 90 100",
        "default_time_grid": "2000 2005 2010",
        "birth_prev": 0,
        "ode_step_size": 1,
        "minimum_meas_cv": 0.1,
        "rate_case": "iota_pos_rho_pos",
        "data_density": "gaussian",
        "constrain_omega": 1,
        "fix_cov": 1,
        "split_sex": 1,
        "modelable_entity_id": 23514,
        "bundle_id": 4325,
        "model_version_id": 267890,
        "add_calc_emr": 1,
        "drill": "drill",
        "drill_sex": 2,
        "drill_location_start": 1,
        "drill_location_end": 3,
        "zero_sum_random": [
            "iota",
            "rho",
            "chi"
        ],
        "quasi_fixed": 0,
        "addl_ode_stpes": "0.015625 0.03125 0.0625 0.125 0.25 0.5",
        "title": "LBW / running on NAME",
        "drill_location": 137
    },
    "max_num_iter": {
        "fixed": 100,
        "random": 100
    },
    "print_level": {
        "fixed": 5,
        "random": 0
    },
    "accept_after_max_steps": {
        "fixed": 5,
        "random": 5
    },
    "students_dof": {
        "priors": 5,
        "data": 5
    },
    "log_students_dof": {
        "priors": 5,
        "data": 5
    },
    "eta": {
        "priors": 0.01,
        "data": 0.01
    },
    "config_version": 1,
    "gbd_round_id": 5,
    "csmr_cod_output_version_id": 90,
    "csmr_mortality_output_version_id": 90,
    "location_set_version_id": 429,
    "min_cv": [
        {
            "cascade_level_id": 1,
            "value": 0.1
        }
    ],
    "tolerance": {
        "fixed": 1e-10
    }
}


def grid(non_negative, rng, single_age=False):
    grid_case = {
        "default": {
            "value": {
                "density": "gaussian",
                "mean": 0.1,
                "std": 0.1
            },
            "dage": {
                "density": "gaussian",
                "mean": 0,
                "std": 0.1
            },
            "dtime": {
                "density": "gaussian",
                "mean": 0,
                "std": 0.1
            }
        },
        "age_time_specific": rng.choice([0, 1], p=[0.25, 0.75]),
    }
    if non_negative:
        # The min has to be > 0?
        grid_case["default"]["value"]["min"] = rng.choice([1e-4])
        grid_case["default"]["value"]["mean"] = grid_case["default"]["value"]["min"] + 1e-2
    age_choice = rng.choice([0, 1, 2])
    if age_choice == 1:
        grid_case["age_grid"] = "0"
    elif age_choice == 2:
        grid_case["age_grid"] = "0 1 5 10"
    if single_age:
        grid_case["age_grid"] = "0"
    time_choice = rng.choice([0, 1, 2])
    if time_choice == 1:
        grid_case["time_grid"] = "2000"
    elif time_choice == 2:
        grid_case["time_grid"] = "2000 2005 2010"
    return grid_case


def rate_grid(rate, rng):
    single_age = rate == "pini"
    grid_case = grid(True, rng, single_age)
    grid_case["rate"] = rate
    return grid_case


def covariate(study_country, covariate_idx, rng):
    assert study_country in ["study", "country"]
    grid_case = grid(False, rng)
    grid_case["mulcov_type"] = rng.choice(["meas_std", "meas_value", "rate_value"])
    grid_case["measure_id"] = 41
    grid_case[f"{study_country}_covariate_id"] = covariate_idx
    grid_case["transformation"] = 0
    return grid_case


def create_settings(rng):
    """
    Makes a random settings, as though EpiViz-AT made it.

    Args:
        rng (numpy.random.RandomState): Create with ``RandomState(2324234)``.

    Returns:
        dict: A dictionary that looks like parsed JSON from EpiViz-AT.
    """
    has = dict()
    for rate, likely in [("iota", 0.1), ("rho", 0.7), ("omega", 0), ("chi", 0.1), ("pini", 0.7)]:
        has[rate] = rng.uniform() > likely

    # The parent location id is 1
    # Would you ever have one child random effect?
    child_selection = [None, [2], [2, 3, 4]]
    children = child_selection[rng.choice([0, 1, 2], p=[0.2, 0.05, 0.75])]

    covariates = [1604, 2453, 6497]

    case = deepcopy(BASE_CASE)
    if has["pini"]:
        case["model"]["birth_prev"] = 1

    case["model"]["add_calc_emr"] = rng.choice([0, 1])
    case["model"]["constrain_omega"] = rng.choice([0, 1], p=[0.1, 0.9])

    # Guarantee at least one nonzero rate, after constraint of omega.
    if case["model"]["constrain_omega"] == 1:
        has["omega"] = True
        if not any(has[x] for x in ["iota", "rho", "chi"]):
            has["iota"] = True
    else:
        if sum(has.values()) == 0:
            has["omega"] = True

    case["rate"] = list()
    for rate, exists in has.items():
        if exists:
            case["rate"].append(rate_grid(rate, rng))

    random_effects = add_random_effects(children, has, rate, rng)

    # Only add to dict if there are some?
    if random_effects:
        case["random_effect"] = random_effects

    add_covariates(case, covariates, rng)
    try:
        config = json_settings_to_frozen_settings(case, 267890)
    except SettingsError:
        pprint(case, indent=2)
        raise
    return config


def add_covariates(case, covariates, rng):
    for ckind in ["study", "country"]:
        scovariates = list()
        for cov_idx in range(rng.randint(4)):
            scovariates.append(covariate(ckind, covariates[cov_idx], rng))
        if scovariates:
            case[f"{ckind}_covariate"] = scovariates


def add_random_effects(children, has, rate, rng):
    random_effects = list()
    if children:
        for random_effect, exists in has.items():
            single_age = rate == "pini"
            choice = rng.choice(["none", "all", "every"])
            if choice == "all":
                grid_case = grid(False, rng, single_age)
                grid_case["rate"] = random_effect
                grid_case["location"] = 1
                random_effects.append(grid_case)
            elif choice == "every":
                for child in children:
                    grid_case = grid(False, rng, single_age)
                    grid_case["rate"] = random_effect
                    grid_case["location"] = child
                    random_effects.append(grid_case)
    return random_effects


def create_local_settings(rng):
    """Make a local settings object, all the way from the EpiViz-AT form."""
    args = parse_arguments(["z.db"])
    locations = nx.DiGraph()
    locations.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5)])
    settings = create_settings(rng)
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    j = list(c.cascade_jobs)
    job_kind, job_args = c.cascade_job(j[rng.randint(len(j))])
    assert job_kind == "estimate_location"
    return job_args
