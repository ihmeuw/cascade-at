BASE_CASE = {
    'model': {
        'random_seed': 551086213,
        'default_age_grid': '0 1 5 10 20 30 40 50 60 70 80 90 100',
        'default_time_grid': '1990 1995 2000 2005 2010 2015 2016',
        'add_calc_emr': 'from_both',
        'birth_prev': 0,
        'ode_step_size': 5,
        'minimum_meas_cv': 0,
        'rate_case': 'iota_pos_rho_zero',
        'data_density': 'log_gaussian',
        'constrain_omega': 0,
        'modelable_entity_id': 2005,
        'decomp_step_id': 3,
        'research_area': 2,
        'title': 'Test',
        'description': '<p>base case test<\\/p>',
        'bundle_id': 173,
        'crosswalk_version_id': 5699,
        'add_csmr_cause': 587,
        'model_version_id': 395831
    },
    'max_num_iter': {
        'fixed': 100,
        'random': 100
    },
    'print_level': {
        'fixed': 5,
        'random': 0},
    'accept_after_max_steps': {
        'fixed': 5,
        'random': 5
    },
    'students_dof': {
        'priors': 5,
        'data': 5},
    'log_students_dof': {
        'priors': 5,
        'data': 5
    },
    'eta': {
        'priors': 0.01,
        'data': 0.01
    },
    'config_version': 'x',
    'min_cv': [{'cascade_level_id': 'most_detailed', 'value': 0.4}],
    'rate': [{'age_time_specific': 1,
              'default': {'value': {'density': 'gaussian'}},
              'rate': 'iota'}],
    'random_effect': [{'age_time_specific': 0}],
    'study_covariate': [{'age_time_specific': 0}],
    'country_covariate': [{'age_time_specific': 0,
                           'mulcov_type': 'meas_value',
                           'country_covariate_id': 28,
                           'measure_id': 6}],
    'gbd_round_id': 6,
    'csmr_cod_output_version_id': 84,
    'csmr_mortality_output_version_id': 8003,
    'location_set_version_id': 544}
