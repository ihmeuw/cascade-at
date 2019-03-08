import asyncio
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from timeit import default_timer as timer
from types import SimpleNamespace

import pandas as pd
from numpy import nan

from cascade.core import getLoggers
from cascade.core.db import dataframe_from_disk, db_queries, age_spans
from cascade.executor.construct_model import construct_model
from cascade.executor.covariate_data import find_covariate_names, add_covariate_data_to_observations_and_avgints
from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.priors_from_draws import set_priors_from_parent_draws
from cascade.executor.session_options import make_options
from cascade.input_data.configuration.construct_bundle import (
    normalized_bundle_from_database,
    normalized_bundle_from_disk,
    bundle_to_observations,
    strip_bundle_exclusions,
)
from cascade.input_data.configuration.construct_country import (
    convert_gbd_ids_to_dismod_values, check_binary_covariates
)
from cascade.input_data.configuration.construct_mortality import get_raw_csmr, normalize_csmr
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.input_data.configuration.local_cache import LocalCache
from cascade.input_data.configuration.raw_input import validate_input_data_types
from cascade.input_data.db.asdr import asdr_as_fit_input
from cascade.input_data.db.country_covariates import country_covariate_set
from cascade.input_data.db.locations import location_hierarchy, location_hierarchy_to_dataframe
from cascade.input_data.db.study_covariates import get_study_covariates
from cascade.model.integrands import make_average_integrand_cases_from_gbd
from cascade.model.session import Session

CODELOG, MATHLOG = getLoggers(__name__)


def estimate_location(execution_context, local_settings, local_cache=None):
    """
    Estimates rates for a single location in the location hierarchy.
    This does multiple fits and predictions in order to estimate uncertainty.

    Args:
        execution_context: Describes environment for this process.
        local_settings: A dictionary describing the work to do. This has
            a location ID corresponding to the location for this fit.
    """
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    input_data = retrieve_data(execution_context, local_settings, covariate_data_spec, local_cache)
    columns_wrong = validate_input_data_types(input_data)
    assert not columns_wrong, f"validation failed {columns_wrong}"
    modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
    model = construct_model(modified_data, local_settings, covariate_multipliers,
                            covariate_data_spec)
    set_priors_from_parent_draws(model, input_data.draws)
    computed_fit, draws = compute_location(execution_context, local_settings, modified_data, model)
    if local_cache:
        local_cache.set(f"fit-draws:{local_settings.parent_location_id}", draws)
    if not local_settings.run.no_upload:
        save_outputs(computed_fit, draws, execution_context, local_settings)


def retrieve_data(execution_context, local_settings, covariate_data_spec, local_cache=None):
    """Gets data from the outside world."""
    local_cache = local_cache if local_cache else LocalCache()
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
    # Raw country covariate data.
    covariates_by_age_id = country_covariate_set(
        country_covariate_ids,
        demographics=dict(age_group_ids="all", year_ids="all", sex_ids="all",
                          location_ids=local_settings.parent_location_id),
        gbd_round_id=data_access.gbd_round_id,
    )
    # Every age group defined, so that we can search for what's given.
    all_age_spans = age_spans.get_age_spans()
    data.country_covariates = dict()
    for covariate_id, covariate_df in covariates_by_age_id.items():
        ccov_ranges_df = convert_gbd_ids_to_dismod_values(covariate_df, all_age_spans)
        data.country_covariates[covariate_id] = ccov_ranges_df

    data.country_covariate_binary = check_binary_covariates(execution_context, country_covariate_ids)

    # Standard GBD age groups with IDs, start, finish.
    data.ages_df = db_queries.get_age_metadata(
        age_group_set_id=data_access.age_group_set_id,
        gbd_round_id=data_access.gbd_round_id
    )
    data.years_df = db_queries.get_demographics(
        gbd_team="epi", gbd_round_id=data_access.gbd_round_id)["year_id"]

    include_birth_prevalence = local_settings.settings.model.birth_prev
    data.average_integrand_cases = \
        make_average_integrand_cases_from_gbd(
            data.ages_df, data.years_df, local_settings.sex_id,
            local_settings.parent_location_id, include_birth_prevalence)
    # This comes in yearly from 1950 to 2018
    data.age_specific_death_rate = asdr_as_fit_input(
        local_settings.parent_location_id, local_settings.sex_id,
        data_access.gbd_round_id, data.ages_df, with_hiv=data_access.with_hiv)

    data.cause_specific_mortality_rate = get_raw_csmr(
        execution_context, local_settings.data_access, local_settings.parent_location_id, all_age_spans)

    data.study_id_to_name, data.country_id_to_name = find_covariate_names(
        execution_context, covariate_data_spec)
    # These are the draws as output of the parent location.
    data.draws = local_cache.get(f"fit-draws:{local_settings.grandparent_location_id}")

    # The parent can also supply integrands as a kind of prior.
    # These will be shaped like input measurement data.
    data.integrands = local_cache.get(f"fit-integrands:{local_settings.grandparent_location_id}")

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

    csmr = normalize_csmr(input_data.cause_specific_mortality_rate, local_settings.sex_id)
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
    add_covariate_data_to_observations_and_avgints(input_data, local_settings, covariate_data_spec)
    input_data.observations = input_data.observations.drop(columns=["sex_id", "seq"])
    set_sex_reference(covariate_data_spec, local_settings)
    return input_data


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
        reference, max_difference = sex_assignments_to_exclude_by_value[tuple(sorted(local_settings.sex_id))]
        sex_covariate[0].reference = reference
        sex_covariate[0].max_difference = max_difference


def compute_location(execution_context, local_settings, input_data, model):
    """

    Args:
        execution_context:
        input_data: These include observations and initial guess.
        model (Model): A complete Model object.

    Returns:
        The fit and draws.
    """
    base_path = execution_context.db_path(local_settings.parent_location_id)
    base_path.mkdir(parents=True, exist_ok=True)
    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename=base_path / "fit.db"
    )
    session.set_option(**make_options(local_settings.settings))
    begin = timer()
    # This should just call init.
    if not local_settings.run.db_only:
        fit_result = session.fit(model, input_data.observations)
    else:
        session.setup_model_for_fit(model, input_data.observations)
        return None, None
    CODELOG.info(f"fit {timer() - begin} success {fit_result.success}")
    draws = make_draws(
        execution_context,
        model,
        input_data,
        fit_result.fit,
        local_settings,
        execution_context.parameters.num_processes
    )
    return fit_result.fit, draws


def save_outputs(computed_fit, draws, execution_context, local_settings):
    return None


def _fit_and_predict_fixed_effect_sample(sim_model, sim_data, fit_file, locations,
                                         parent_location, local_settings, draw_idx):
    sim_session = Session(
        locations=locations,
        parent_location=parent_location,
        filename=fit_file
    )
    local_settings.settings.policies.meas_std_effect
    sim_session.set_option(**make_options(local_settings.settings))
    begin = timer()
    sim_fit_result = sim_session.fit(sim_model, sim_data)
    CODELOG.info(f"fit {timer() - begin} success {sim_fit_result.success}")
    if sim_fit_result.success:
        CODELOG.debug(f"sim fit {draw_idx} success")
        return sim_fit_result.fit
    else:
        CODELOG.debug(f"sim fit {draw_idx} not successful in {fit_file}.")
        return None
    # XXX make the Session close or be a contextmanager.


async def _async_make_draws(base_path, input_data, model, local_settings, simulate_result, num_processes):
    jobs = list()

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(num_processes) as pool:
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
                local_settings,
                draw_idx
            ))

        results = await asyncio.gather(*jobs)
    return [r for r in results if r is not None]


def make_draws(execution_context, model, input_data, max_fit, local_settings, num_processes):
    base_path = execution_context.db_path(local_settings.parent_location_id)
    base_path.mkdir(parents=True, exist_ok=True)
    draw_cnt = local_settings.number_of_fixed_effect_samples
    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename=base_path / "simulate.db"
    )
    session.set_option(**make_options(local_settings.settings))
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
