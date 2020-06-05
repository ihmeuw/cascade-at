BASE_CASE = {
    "model": {
        "random_seed": 495279142,
        "default_age_grid": "0 0.01917808 0.07671233 1 5 10 20 30 40 50 60 70 80 90 100",
        "default_time_grid": "1990 1995 2000 2005 2010 2015 2016",
        "add_calc_emr": "from_both",
        "birth_prev": 0,
        "ode_step_size": 5,
        "minimum_meas_cv": 0.2,
        "rate_case": "iota_pos_rho_zero",
        "data_density": "log_gaussian",
        "constrain_omega": 1,
        "modelable_entity_id": 2005,
        "decomp_step_id": 3,
        "research_area": 2,
        "drill": "drill",
        "drill_location_start": 70,
        "bundle_id": 173,
        "crosswalk_version_id": 5699,
        "split_sex": "most_detailed",
        "add_csmr_cause": 587,
        "drill_sex": 2,
        "model_version_id": 472515,
        "title": "test diabetes australasia marlena -- 2",
        "relabel_incidence": 2,
        "description": "<p>diabetes<\/p>",
        "addl_ode_stpes": "0.01917808 0.07671233 1.0",
        "zero_sum_random": [
            "iota"
        ],
        "bound_frac_fixed": 1.0e-8,
        "drill_location_end": [72],
        "quasi_fixed": 0
    },
    "max_num_iter": {
        "fixed": 200,
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
        "priors": 1.0e-5,
        "data": 1.0e-5
    },
    "config_version": "mnorwood",
    "rate": [
        {
            "age_time_specific": 1,
            "default": {
                "value": {
                    "density": "gaussian",
                    "min": 1.0e-6,
                    "mean": 0.00015,
                    "max": 0.01,
                    "std": 1.5,
                    "eta": 1.0e-6
                },
                "dage": {
                    "density": "gaussian",
                    "min": -1,
                    "mean": 0,
                    "max": 1,
                    "std": 0.01
                },
                "dtime": {
                    "density": "gaussian",
                    "min": -1,
                    "mean": 0,
                    "max": 1,
                    "std": 0.01
                }
            },
            "rate": "iota",
            "age_grid": "0 5 10 50 100"
        },
        {
            "age_time_specific": 1,
            "default": {
                "value": {
                    "density": "gaussian",
                    "min": 1.0e-6,
                    "mean": 0.0004,
                    "max": 0.01,
                    "std": 0.2
                },
                "dage": {
                    "density": "gaussian",
                    "min": -1,
                    "mean": 0,
                    "max": 1,
                    "std": 0.01
                },
                "dtime": {
                    "density": "gaussian",
                    "min": -1,
                    "mean": 0,
                    "max": 1,
                    "std": 0.01
                }
            },
            "rate": "chi"
        },
        {
            "age_time_specific": 0,
            "default": {
                "value": {
                    "density": "log_gaussian",
                    "min": 0,
                    "mean": 0.1,
                    "max": 0.2,
                    "std": 1,
                    "eta": 1.0e-6
                },
                "dage": {
                    "density": "uniform",
                    "min": -1,
                    "mean": 0,
                    "max": 1
                },
                "dtime": {
                    "density": "uniform",
                    "min": -1,
                    "mean": 0,
                    "max": 1
                }
            },
            "rate": "pini"
        }
    ],
    "random_effect": [
        {
            "age_time_specific": 0,
            "default": {
                "value": {
                    "density": "gaussian",
                    "mean": 0,
                    "std": 1
                },
                "dage": {
                    "mean": 0,
                    "std": 1,
                    "density": "uniform"
                },
                "dtime": {
                    "mean": 0,
                    "std": 1,
                    "density": "uniform"
                }
            },
            "rate": "iota"
        }
    ],
    "study_covariate": [
        {
            "age_time_specific": 0,
            "mulcov_type": "rate_value",
            "default": {
                "value": {
                    "density": "uniform",
                    "min": -1,
                    "mean": 0,
                    "max": 1
                },
                "dage": {
                    "density": "uniform",
                    "min": -1,
                    "mean": 0,
                    "max": 1
                },
                "dtime": {
                    "density": "uniform",
                    "min": -1,
                    "mean": 0,
                    "max": 1
                }
            },
            "study_covariate_id": 0,
            "transformation": 0,
            "measure_id": 41
        }
    ],
    "country_covariate": [
        {
            "age_time_specific": 0,
            "mulcov_type": "rate_value",
            "measure_id": 41,
            "country_covariate_id": 28,
            "transformation": 0,
            "default": {
                "value": {
                    "density": "uniform",
                    "min": -1,
                    "mean": 0,
                    "max": 1,
                    "eta": 1.0e-5
                }
            }
        }
    ],
    "gbd_round_id": 6,
    "csmr_cod_output_version_id": 84,
    "csmr_mortality_output_version_id": 8003,
    "location_set_version_id": 544,
    "tolerance": {
        "fixed": 1.0e-6,
        "random": 1.0e-6
    }
}
