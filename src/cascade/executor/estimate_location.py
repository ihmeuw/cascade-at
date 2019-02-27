from collections import defaultdict
from timeit import default_timer as timer
from types import SimpleNamespace

from numpy import nan

from cascade.core import getLoggers
from cascade.core.db import dataframe_from_disk
from cascade.core.db import db_queries
from cascade.executor.construct_model import construct_model
from cascade.executor.covariate_description import create_covariate_specifications
from cascade.executor.priors_from_draws import set_priors_from_parent_draws
from cascade.executor.session_options import make_options
from cascade.input_data.configuration.construct_bundle import (
    normalized_bundle_from_database,
    normalized_bundle_from_disk,
    bundle_to_observations
)
from cascade.model.integrands import make_average_integrand_cases_from_gbd
from cascade.input_data.configuration.id_map import make_integrand_map
from cascade.input_data.db.asdr import asdr_as_fit_input
from cascade.input_data.db.locations import location_hierarchy, location_hierarchy_to_dataframe
from cascade.input_data.db.study_covariates import get_study_covariates
from cascade.model.session import Session

CODELOG, MATHLOG = getLoggers(__name__)


def estimate_location(execution_context, local_settings):
    """
    Estimates rates for a single location in the location hierarchy.
    This does multiple fits and predictions in order to estimate uncertainty.

    Args:
        execution_context: Describes environment for this process.
        local_settings: A dictionary describing the work to do. This has
            a location ID corresponding to the location for this fit.
    """
    input_data = retrieve_data(execution_context, local_settings)
    covariate_multipliers, covariate_data_spec = create_covariate_specifications(
        local_settings.settings.country_covariate, local_settings.settings.study_covariate
    )
    modified_data = modify_input_data(input_data, local_settings, covariate_data_spec)
    model = construct_model(modified_data, local_settings, covariate_multipliers)
    set_priors_from_parent_draws(model, input_data.draws)
    computed_fit, draws = compute_location(execution_context, local_settings, modified_data, model)
    save_outputs(computed_fit, draws, execution_context, local_settings)


def retrieve_data(execution_context, local_settings, covariate_data_spec):
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

    # These are the draws as output of the parent location.
    data.draws = None

    # The parent can also supply integrands as a kind of prior.
    # These will be shaped like input measurement data.
    data.integrands = None

    return data


def modify_input_data(input_data, local_settings, covariate_data_spec):
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

    input_data.observations = bundle_to_observations(
        input_data.bundle,
        local_settings.parent_location_id,
        data_eta,
        density,
    )
    input_data.observations = input_data.observations.drop(columns="sex_id")
    # ev_settings.data_eta_by_integrand is a dummy in form.py.
    MATHLOG.info(f"Ignoring data_eta_by_integrand")

    input_data.locations_df = location_hierarchy_to_dataframe(input_data.locations)
    return input_data


def compute_location(execution_context, local_settings, input_data, model):
    """

    Args:
        execution_context:
        input_data: These include observations and initial guess.
        model (Model): A complete Model object.

    Returns:
        The fit.
    """

    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename="subsession.db"
    )
    session.set_option(**make_options(local_settings.settings))
    begin = timer()
    # This should just call init.
    fit_result = session.fit(model, input_data.observations)
    CODELOG.info(f"fit {timer() - begin} success {fit_result.success}")
    draws = make_draws(model, input_data, fit_result.fit, local_settings)
    return fit_result.fit, draws


def save_outputs(computed_fit, draws, execution_context, local_settings):
    return None


def make_draws(model, input_data, max_fit, local_settings):
    draw_cnt = local_settings.number_of_fixed_effect_samples
    session = Session(
        locations=input_data.locations_df,
        parent_location=model.location_id,
        filename="simulate.db"
    )
    session.set_option(**make_options(local_settings.settings))
    simulate_result = session.simulate(model, input_data.observations, max_fit, draw_cnt)

    draws = list()
    for draw_idx in range(simulate_result.count):
        sim_model, sim_data = simulate_result.simulation(draw_idx)
        # let's start a new session because the simulation results are associated
        # with a session and running a new fit will delete them.
        fit_file = f"simulate{draw_idx}.db"
        sim_session = Session(
            locations=input_data.locations_df,
            parent_location=model.location_id,
            filename=fit_file
        )
        sim_session.set_option(**make_options(local_settings.settings))
        begin = timer()
        sim_fit_result = sim_session.fit(sim_model, sim_data)
        CODELOG.info(f"fit {timer() - begin} success {sim_fit_result.success}")
        if sim_fit_result.success:
            draws.append(sim_fit_result.fit)
            print(f"sim fit {draw_idx} success")
        else:
            print(f"sim fit {draw_idx} not successful in {fit_file}.")
        # XXX make the Session close or be a contextmanager.
        del sim_session
    return draws
