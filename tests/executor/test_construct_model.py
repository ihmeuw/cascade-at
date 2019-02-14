"""
Test model construction. Variation in models has these dimensions:

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


BASE_CASE = {
    "model": {
        "random_seed": 899736881,
        "default_age_grid": "0 .1 .2 1 5 10 20 30 40 50 60 70 80 90 100",
        "default_time_grid": "2000, 2005, 2010",
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
        "drill_location_end": 31,
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
    "rate": [
        {
            "default": {
                "value": {
                    "density": "uniform",
                    "min": 1e-10,
                    "mean": 0.019,
                    "max": 0.2
                },
                "dage": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.01
                }
            },
            "rate": "rho",
            "age_time_specific": 1
        },
        {
            "default": {
                "value": {
                    "density": "uniform",
                    "min": 1e-10,
                    "mean": 0.01,
                    "max": 0.04
                },
                "dage": {
                    "std": 0.002,
                    "density": "gaussian",
                    "mean": 0
                }
            },
            "age_time_specific": 1,
            "rate": "iota"
        },
        {
            "default": {
                "value": {
                    "density": "uniform",
                    "min": 1e-10,
                    "mean": 0.1,
                    "max": 0.2
                },
                "dage": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.02
                }
            },
            "rate": "chi",
            "age_time_specific": 1
        }
    ],
    "min_cv": [
        {
            "cascade_level_id": 1,
            "value": 0.1
        }
    ],
    "random_effect": [
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.1
                }
            },
            "rate": "iota",
            "age_time_specific": 0,
            "location": 1,
            "age_grid": 0,
            "time_grid": 2000
        },
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.1
                }
            },
            "rate": "rho",
            "age_time_specific": 0,
            "location": 1,
            "age_grid": 0,
            "time_grid": 2000
        },
        {
            "default": {
                "value": {
                    "mean": 0,
                    "density": "gaussian",
                    "std": 0.1
                }
            },
            "rate": "chi",
            "age_time_specific": 0,
            "location": 1,
            "age_grid": 0,
            "time_grid": 2000
        }
    ],
    "study_covariate": [
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.0001,
                    "min": 0,
                    "max": 0.1
                }
            },
            "mulcov_type": "meas_std",
            "measure_id": 41,
            "study_covariate_id": 1604,
            "transformation": 0,
            "age_grid": 0,
            "time_grid": "2000, 2005, 2010",
            "age_time_specific": 1
        },
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.0001,
                    "min": 0,
                    "max": 0.1
                }
            },
            "mulcov_type": "meas_std",
            "measure_id": 7,
            "study_covariate_id": 1604,
            "transformation": 0,
            "age_grid": 0,
            "time_grid": "2000, 2005, 2010",
            "age_time_specific": 1
        },
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.0001,
                    "min": 0,
                    "max": 0.1
                }
            },
            "mulcov_type": "meas_std",
            "measure_id": 9,
            "study_covariate_id": 1604,
            "transformation": 0,
            "age_grid": 0,
            "time_grid": "2000, 2005, 2010",
            "age_time_specific": 1
        },
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.0001,
                    "min": 0,
                    "max": 0.1
                }
            },
            "mulcov_type": "meas_std",
            "measure_id": 5,
            "study_covariate_id": 1604,
            "transformation": 0,
            "age_grid": 0,
            "time_grid": "2000, 2005, 2010",
            "age_time_specific": 1
        },
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 0.0001,
                    "min": 0,
                    "max": 0.1
                }
            },
            "mulcov_type": "meas_std",
            "measure_id": 15,
            "study_covariate_id": 1604,
            "time_grid": "2000, 2005, 2010",
            "age_grid": 0,
            "transformation": 0,
            "age_time_specific": 1
        }
    ],
    "tolerance": {
        "fixed": 1e-10
    }
}


def rate_grid(rng):

    grid_case = {
        "default": {
            "value": {
                "density": "gaussian",
                "mean": 0,
                "std": 0.1
            }
        },
        "rate": "iota",
        "age_time_specific": 0,
        "location": 1,
        "age_grid": 0,
        "time_grid": 2000
    }
    return grid_case


def create_model(rng):
    has = dict()
    for rate, likely in [("iota", 0.1), ("rho", 0.7), ("omega", 0), ("chi", 0.1), ("pini", 0.7)]:
        has[rate] = rng.uniform() > likely

    # Would you ever have one child random effect?
    child_selection = [None, [2], [2, 3, 4]]
    children = child_selection[rng.choice([0, 1, 2], p=[0.2, 0.05, 0.75])]

    cov_select = [None, ["traffic", "smoking", "one"]]
    covariates = cov_select[rng.choice([0, 1], p=[0.3, 0.8])]

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

    assert children
    assert covariates
