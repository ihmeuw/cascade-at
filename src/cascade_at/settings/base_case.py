BASE_CASE = {
  "model": {
    "random_seed": 290118279,
    "default_age_grid": "0 1 5 10 20 30 40 50 60 70 80 90 100",
    "default_time_grid": "1990 1995 2000 2005 2010 2015 2016",
    "add_calc_emr": "from_both",
    "birth_prev": 0,
    "ode_step_size": 5,
    "minimum_meas_cv": 0,
    "rate_case": "iota_pos_rho_zero",
    "data_density": "log_gaussian",
    "constrain_omega": 0,
    "modelable_entity_id": 2005,
    "decomp_step_id": 3,
    "research_area": 2,
    "title": "Test model",
    "description": "<p>test</p>",
    "bundle_id": 173,
    "crosswalk_version_id": 5699,
    "add_csmr_cause": 587,
    "zero_sum_random": [
      "rho",
      "iota"
    ],
    "quasi_fixed": 0
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
  "min_cv": [
    {
      "cascade_level_id": "most_detailed",
      "value": 0.4
    }
  ],
  "min_cv_by_rate": [
    {
      "rate_measure_id": "iota",
      "cascade_level_id": "most_detailed",
      "value": 0.4
    }
  ],
  "study_covariate": [
    {
      "age_time_specific": 0,
      "mulcov_type": "meas_value",
      "measure_id": 6
    }
  ],
  "country_covariate": [
    {
      "age_time_specific": 0,
      "default": {
        "value": {
          "density": "gaussian",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        },
        "dage": {
          "min": 1,
          "density": "uniform",
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        },
        "dtime": {
          "min": 1,
          "density": "laplace",
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        }
      },
      "mulstd": {
        "value": {
          "density": "laplace",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        },
        "dage": {
          "density": "laplace",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        },
        "dtime": {
          "density": "laplace",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        }
      },
      "detail": [
        {
          "age_lower": 1,
          "age_upper": -1,
          "prior_type": "dage",
          "time_lower": 1,
          "time_upper": 1,
          "born_lower": 1,
          "born_upper": 1,
          "density": "log_laplace",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": -1,
          "eta": -1
        },
        {
          "prior_type": "dtime",
          "age_lower": 1,
          "age_upper": -1,
          "time_upper": -1,
          "time_lower": -1,
          "born_lower": -1,
          "born_upper": -1,
          "density": "log_laplace",
          "min": -1,
          "mean": -1,
          "max": -1,
          "std": -1,
          "eta": 2
        },
        {
          "prior_type": "value",
          "age_lower": -1,
          "age_upper": -1,
          "time_lower": -1,
          "time_upper": -2,
          "born_lower": -1,
          "born_upper": 0,
          "density": "students",
          "min": -1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        }
      ],
      "mulcov_type": "rate_value",
      "measure_id": 6,
      "country_covariate_id": 28,
      "transformation": 1
    }
  ],
  "rate": [
    {
      "age_time_specific": 1,
      "default": {
        "value": {
          "density": "gaussian"
        }
      },
      "mulstd": {
        "value": {
          "density": "students"
        }
      },
      "rate": "iota",
      "age_grid": "0 1 5 10 20 30 40 50 60 70 80 90 100",
      "time_grid": "1990 1995 2000 2005 2010 2015 2016"
    }
  ],
  "random_effect": [
    {
      "age_time_specific": 1,
      "default": {
        "value": {
          "density": "gaussian",
          "mean": 1,
          "std": 1,
          "eta": 1
        },
        "dage": {
          "density": "uniform",
          "eta": 1,
          "std": 1,
          "mean": 1
        },
        "dtime": {
          "density": "uniform",
          "std": 1,
          "eta": 1,
          "mean": 1
        }
      },
      "mulstd": {
        "value": {
          "density": "gaussian",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        },
        "dage": {
          "density": "uniform",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        },
        "dtime": {
          "density": "uniform",
          "min": 1,
          "mean": 1,
          "max": 1,
          "std": 1,
          "eta": 1
        }
      },
      "detail": [
        {
          "prior_type": "value",
          "age_lower": 1,
          "age_upper": 1,
          "time_lower": 0,
          "time_upper": 1,
          "born_lower": 1,
          "born_upper": 1,
          "density": "log_gaussian",
          "mean": 1,
          "std": 1,
          "eta": 1
        },
        {
          "prior_type": "dage",
          "age_lower": 1,
          "age_upper": 1,
          "time_lower": 1,
          "time_upper": 1,
          "born_lower": 1,
          "born_upper": 1,
          "density": "laplace",
          "mean": 1,
          "std": 1,
          "eta": 1
        },
        {
          "prior_type": "dtime",
          "age_lower": 1,
          "age_upper": 1,
          "time_lower": 1,
          "time_upper": 1,
          "born_lower": 1,
          "born_upper": 1,
          "density": "log_students",
          "mean": 1,
          "std": -1,
          "eta": 1
        }
      ],
      "rate": "iota",
      "age_grid": "0 10 100",
      "time_grid": "1990 2000 2006"
    }
  ],
  "tolerance": {
    "fixed": 3,
    "random": -3
  },
}
