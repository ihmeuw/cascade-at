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
from numpy.random import RandomState

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
        "drill_location_end": 40,
        "zero_sum_random": [
            "iota",
            "rho",
            "chi"
        ],
        "quasi_fixed": 0,
        "addl_ode_stpes": "0.015625 0.03125 0.0625 0.125 0.25 0.5",
        "title": "LBW / running on NAME",
        "drill_location": 40
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
    # else 0 use default ages
    if single_age:
        grid_case["age_grid"] = "0"
    time_choice = rng.choice([0, 1, 2])
    if time_choice == 1:
        grid_case["time_grid"] = "2000"
    elif time_choice == 2:
        grid_case["time_grid"] = "2000 2005 2010"
    # else 0 use default times
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


def create_settings(rng, locations):
    """
    Makes a random settings, as though EpiViz-AT made it.

    Args:
        rng (numpy.random.RandomState): Create with ``RandomState(2324234)``.
        locations (nx.DiGraph): List of locations.

    Returns:
        dict: A dictionary that looks like parsed JSON from EpiViz-AT.
    """
    has = dict()
    for rate, likely in [("iota", 0.1), ("rho", 0.7), ("omega", 0), ("chi", 0.1), ("pini", 0.7)]:
        has[rate] = rng.uniform() > likely

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

    location_root = 1
    # Use the last location as the drill end possibility.
    last_location = list(nx.topological_sort(locations))[-1]
    drill = sorted(nx.ancestors(locations, last_location))
    start_loc_idx = rng.randint(0, len(drill))
    end_loc_idx = rng.randint(start_loc_idx, len(drill))
    start_loc = drill[start_loc_idx]
    end_loc = drill[end_loc_idx]
    case["model"]["drill_location_start"] = start_loc
    case["model"]["drill_location_end"] = end_loc

    # We specify priors of random effects either
    # a) Once for the whole hierarchy, or
    # b) For every single location in the hierarchy.
    # Choose this by rate
    rate_specifies_re_by_location = dict()
    for rate_name in [x for (x, y) in has.items() if y]:
        choice = rng.choice(["none", "all", "every"])
        # none means no random effects.
        # all means every single location.
        # every means each location gets a new one.
        if rate in {"all", "every"}:
            rate_specifies_re_by_location[rate_name] = choice == "every"
        else:
            pass  # Don't record this rate as needing a random effect.

    random_effects = list()
    # Iterates over every set of parent and children in the graph.
    for rate_name, by_location in rate_specifies_re_by_location:
        if by_location:
            # You have to do _all_ of them.
            for parent, children in nx.bfs_successors(locations, location_root):
                random_effects.extend(add_random_effects(children, rate_name, rng))
        else:
            add_random_effects(None, rate_name, rng)

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


def add_random_effects(children, rate_name, rng):
    random_effects = list()
    single_age = rate_name == "pini"
    if children:
        for child in children:
            grid_case = grid(False, rng, single_age)
            grid_case["rate"] = rate_name
            grid_case["location"] = child
            random_effects.append(grid_case)
    else:
        grid_case = grid(False, rng, single_age)
        grid_case["rate"] = rate_name
        grid_case["location"] = None
        random_effects.append(grid_case)
    return random_effects


def make_locations(depth):
    """Creates locations of given depth as a balanced tree. Root is 1."""
    arity = 3
    zero_based = nx.balanced_tree(arity, depth, nx.DiGraph)
    locations = nx.relabel_nodes(zero_based, {i: i + 1 for i in range(len(zero_based))})
    for lidx, n in enumerate(locations.nodes):
        locations.nodes[n]["location_name"] = str(lidx)
    return locations


def create_local_settings(rng=None):
    """Make a local settings object, all the way from the EpiViz-AT form."""
    rng = rng if rng else RandomState(3242352)
    args = parse_arguments(["z.db"])
    depth = 3
    locations = make_locations(depth)
    settings = create_settings(rng, locations)
    c = CascadePlan.from_epiviz_configuration(locations, settings, args)
    j = list(c.cascade_jobs)
    job_kind, job_args = c.cascade_job(j[rng.randint(1, len(j))])
    assert job_kind == "estimate_location"
    return job_args, locations
