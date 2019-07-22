import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from timeit import default_timer as timer
from types import SimpleNamespace

import pandas as pd
from numpy import nan

from cascade.core import getLoggers
from cascade.core.db import db_queries, age_spans
from cascade.dismod import DismodATException
from cascade.executor.covariate_data import find_covariate_names, add_covariate_data_to_observations_and_avgints
from cascade.executor.session_options import make_options, make_minimum_meas_cv
from cascade.input_data.configuration.construct_bundle import (
    normalized_bundle_from_database,
    normalized_bundle_from_disk,
    bundle_to_observations,
    strip_bundle_exclusions,
    dataframe_from_disk)
from cascade.input_data.configuration.construct_country import check_binary_covariates
from cascade.input_data.configuration.construct_country import convert_gbd_ids_to_dismod_values
from cascade.input_data.configuration.construct_mortality import get_raw_csmr, normalize_csmr
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.input_data.db.asdr import asdr_as_fit_input
from cascade.input_data.db.country_covariates import country_covariate_set
from cascade.input_data.db.locations import location_hierarchy, location_hierarchy_to_dataframe
from cascade.input_data.db.study_covariates import get_study_covariates
from cascade.model.integrands import make_average_integrand_cases_from_gbd
from cascade.model.session import Session
from cascade.saver.save_prediction import save_predicted_value, uncertainty_from_prediction_draws

CODELOG, MATHLOG = getLoggers(__name__)


def retrieve_data(execution_context, local_settings, covariate_data_spec):
    """Gets data from the outside world."""
    data = SimpleNamespace()
    data_access = local_settings.data_access
    model_version_id = data_access.model_version_id

    data.locations = location_hierarchy(
        data_access.gbd_round_id, location_set_version_id=data_access.location_set_version_id)

    if data_access.bundle_file:
        data.bundle = normalized_bundle_from_disk(data_access.bundle_file)
    else:
        data.bundle = normalized_bundle_from_database(
            execution_context,
            model_version_id,
            bundle_id=local_settings.data_access.bundle_id,
            tier=local_settings.data_access.tier
        )
    # Study covariates will have columns {"bundle_id", "seq", "study_covariate_id"}.
    if data_access.bundle_study_covariates_file:
        data.sparse_covariate_data = dataframe_from_disk(data_access.bundle_study_covariates_file)
    else:
        mvid = data_access.model_version_id
        data.sparse_covariate_data = get_study_covariates(
            execution_context, data_access.bundle_id, mvid, tier=data_access.tier)

    country_covariate_ids = {spec.covariate_id for spec in covariate_data_spec if spec.study_country == "country"}
    # Raw country covariate data. Must be subset for children.
    covariates_by_age_id = country_covariate_set(
        country_covariate_ids,
        demographics=dict(age_group_ids="all", year_ids="all", sex_ids="all",
                          location_ids=list(data.locations.nodes)),
        gbd_round_id=data_access.gbd_round_id,
        decomp_step=data_access.decomp_step,
    )
    # Every age group defined, so that we can search for what's given.
    all_age_spans = age_spans.get_age_spans()
    data.country_covariates = dict()
    for covariate_id, covariate_df in covariates_by_age_id.items():
        ccov_ranges_df = convert_gbd_ids_to_dismod_values(covariate_df, all_age_spans)
        data.country_covariates[covariate_id] = ccov_ranges_df

    data.country_covariates_binary = check_binary_covariates(execution_context, country_covariate_ids)

    # Standard GBD age groups with IDs, start, finish.
    data.ages_df = db_queries.get_age_metadata(
        age_group_set_id=data_access.age_group_set_id,
        gbd_round_id=data_access.gbd_round_id
    )
    # Returns a dictionary of demographic IDs.
    data.years_df = db_queries.get_demographics(
        gbd_team="epi", gbd_round_id=data_access.gbd_round_id)["year_id"]

    # This comes in yearly from 1950 to 2018
    # Must be subset for children.
    data.age_specific_death_rate = asdr_as_fit_input(
        data_access.location_set_version_id,
        local_settings.sexes,
        data_access.gbd_round_id,
        data_access.decomp_step,
        data.ages_df,
        with_hiv=data_access.with_hiv
    )
    node = next(iter(data.locations.nodes))
    location_set_id = data.locations.nodes[node]["location_set_id"]
    data.cause_specific_mortality_rate = get_raw_csmr(
        execution_context, local_settings.data_access, location_set_id, all_age_spans)

    data.study_id_to_name, data.country_id_to_name = find_covariate_names(
        execution_context, covariate_data_spec)

    return data


def modify_input_data(input_data, local_settings, covariate_data_spec):
    """Transforms data to input for model."""
    ev_settings = local_settings.settings
    # These are suitable for input to the fit.
    if not ev_settings.eta.is_field_unset("data") and ev_settings.eta.data:
        data_eta = defaultdict(lambda: float(ev_settings.eta.data))
    else:
        data_eta = defaultdict(lambda: nan)
    id_to_integrand = make_integrand_map()
    for set_eta in ev_settings.data_eta_by_integrand:
        data_eta[id_to_integrand[set_eta.integrand_measure_id]] = float(set_eta.value)

    if not ev_settings.model.is_field_unset("data_density") and ev_settings.model.data_density:
        density = defaultdict(lambda: ev_settings.model.data_density)
    else:
        density = defaultdict(lambda: "gaussian")
    for set_density in ev_settings.data_density_by_integrand:
        density[id_to_integrand[set_density.integrand_measure_id]] = set_density.value

    csmr = normalize_csmr(input_data.cause_specific_mortality_rate, local_settings.sexes)
    CODELOG.debug(f"bundle cols {input_data.bundle.columns}\ncsmr cols {csmr.columns}")
    assert not set(csmr.columns) - set(input_data.bundle.columns)
    bundle_with_added = pd.concat([input_data.bundle, csmr], sort=False)
    bundle_without_excluded = strip_bundle_exclusions(bundle_with_added, ev_settings)
    nu = defaultdict(lambda: nan)
    nu["students"] = local_settings.settings.students_dof.data
    nu["log_students"] = local_settings.settings.log_students_dof.data

    # These observations still have a seq column.
    input_data.observations = bundle_to_observations(
        bundle_without_excluded,
        local_settings.parent_location_id,
        data_eta,
        density,
        nu,
    )
    # ev_settings.data_eta_by_integrand is a dummy in form.py.
    MATHLOG.info(f"Ignoring data_eta_by_integrand")

    input_data.locations_df = location_hierarchy_to_dataframe(input_data.locations)
    return input_data


def one_location_data_from_global_data(global_data, local_settings):
    # subset asdr
    # subset csmr
    add_covariate_data_to_observations_and_avgints(global_data, local_settings, global_data.covariate_data_spec)
    global_data.observations = global_data.observations.drop(columns=["sex_id", "seq"])
    set_sex_reference(global_data.covariate_data_spec, local_settings)

    # These are the draws as output of the parent location. Called draws.
    global_data.draws = None

    # The parent can also supply integrands as a kind of prior.
    # These will be shaped like input measurement data. Called fit-integrands.
    global_data.integrands = None

    include_birth_prevalence = local_settings.settings.model.birth_prev
    global_data.average_integrand_cases = \
        make_average_integrand_cases_from_gbd(
            global_data.ages_df,
            global_data.years_df,
            local_settings.sexes,
            local_settings.children,
            include_birth_prevalence
        )
    return global_data


def set_sex_reference(covariate_data_spec, local_settings):
    """The sex covariate holds out data for the sex by setting the ``reference``
    and ``max_difference``. If sex is 1, then set reference to 0.5 and max
    difference to 0.75. If sex is 2, reference is -0.5. If it's 3 or 4,
    then reference is 0."""
    sex_covariate = [sc_sex for sc_sex in covariate_data_spec
                     if sc_sex.covariate_id == 0 and sc_sex.transformation_id == 0]
    if sex_covariate:
        sex_assignments_to_exclude_by_value = {
            (1,): [0.5, 0.25],
            (2,): [-0.5, 0.25],
            (3,): [0.0, 0.25],
            (1, 3): [0.5, 0.75],
            (2, 3): [-0.5, 0.75],
            (1, 2, 3): [0.0, 0.75],
        }
        reference, max_difference = sex_assignments_to_exclude_by_value[tuple(sorted(local_settings.sexes))]
        sex_covariate[0].reference = reference
        sex_covariate[0].max_difference = max_difference


def compute_parent_fit_fixed(execution_context, local_settings, input_data, model, initial_guess=None):
    """

    Args:
        execution_context:
        input_data: These include observations and initial guess.
        model (Model): A complete Model object.

    Returns:
        The fit.
    """
    base_path = execution_context.db_path(local_settings.parent_location_id)
    base_path.mkdir(parents=True, exist_ok=True)
    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename=base_path / "fit.db",
    )
    session.set_option(**make_options(local_settings.settings, local_settings.model_options))
    begin = timer()
    fit_result = session.fit(model, input_data.observations, initial_guess=initial_guess)
    CODELOG.info(f"fit fixed {timer() - begin}")
    if not fit_result.success:
        raise DismodATException("Fit fixed failed")
    return fit_result


def compute_parent_fit(execution_context, local_settings, input_data, model, initial_guess=None):
    """

    Args:
        execution_context:
        input_data: These include observations and initial guess.
        model (Model): A complete Model object.

    Returns:
        The fit.
    """
    base_path = execution_context.db_path(local_settings.parent_location_id)
    base_path.mkdir(parents=True, exist_ok=True)
    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename=base_path / "fit.db",
    )
    session.set_option(**make_options(local_settings.settings, local_settings.model_options))
    begin = timer()
    # This should just call init.
    if not local_settings.run.db_only:
        fit_result = session.fit(model, input_data.observations, initial_guess=initial_guess)
        CODELOG.info(f"fit {timer() - begin}")
        if not fit_result.success:
            raise DismodATException("Fit failed")
        return fit_result
    else:
        session.setup_model_for_fit(model, input_data.observations)
        return None


def save_outputs(computed_fit, predictions, execution_context, local_settings):
    predictions = uncertainty_from_prediction_draws(predictions)

    save_predicted_value(execution_context, predictions, "fit")


def _fit_and_predict_fixed_effect_sample(sim_model, sim_data, fit_file, locations,
                                         parent_location, covariates, average_integrand_cases,
                                         local_settings, draw_idx):
    sim_session = Session(
        locations=locations,
        parent_location=parent_location,
        filename=fit_file
    )
    sim_session.set_option(**make_options(local_settings.settings, local_settings.model_options))
    sim_session.set_minimum_meas_cv(**make_minimum_meas_cv(local_settings.settings))
    begin = timer()
    sim_fit_result = sim_session.fit(sim_model, sim_data)
    CODELOG.info(f"fit {timer() - begin} success {sim_fit_result.success}")

    CODELOG.debug(f"sim fit {draw_idx} success")
    predicted, _ = sim_session.predict(
        sim_fit_result.fit,
        average_integrand_cases.drop("sex_id", "columns"),
        parent_location,
        covariates=covariates
    )
    return (sim_fit_result.fit, predicted)
    # XXX make the Session close or be a contextmanager.


async def _async_make_draws(base_path, input_data, model, local_settings, simulate_result, num_processes):
    jobs = list()

    if num_processes == 1:
        executor = ThreadPoolExecutor
    else:
        executor = ProcessPoolExecutor

    loop = asyncio.get_event_loop()
    with executor(num_processes) as pool:
        for draw_idx in range(simulate_result.count):
            fit_file = f"simulate{draw_idx}.db"
            sim_model, sim_data = simulate_result.simulation(draw_idx)
            jobs.append(loop.run_in_executor(
                pool,
                _fit_and_predict_fixed_effect_sample,
                sim_model,
                sim_data,
                base_path / fit_file,
                input_data.locations_df,
                model.location_id,
                model.covariates,
                input_data.average_integrand_cases,
                local_settings,
                draw_idx
            ))

        results = await asyncio.gather(*jobs)
    return [r for r in results if r is not None]


def make_draws(execution_context, model, input_data, max_fit, local_settings, num_processes):
    base_path = execution_context.db_path(local_settings.parent_location_id)
    MATHLOG.info(f"DB files for {local_settings.parent_location_id} in {base_path}.")
    base_path.mkdir(parents=True, exist_ok=True)
    draw_cnt = local_settings.number_of_fixed_effect_samples
    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename=base_path / "simulate.db"
    )
    session.set_option(**make_options(local_settings.settings, local_settings.model_options))
    session.set_minimum_meas_cv(**make_minimum_meas_cv(local_settings.settings))
    simulate_result = session.simulate(model, input_data.observations, max_fit, draw_cnt)

    loop = asyncio.get_event_loop()
    draws = loop.run_until_complete(
        _async_make_draws(
            base_path,
            input_data,
            model,
            local_settings,
            simulate_result,
            num_processes,
        )
    )

    return draws
