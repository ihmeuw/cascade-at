from types import SimpleNamespace

from cascade.core import getLoggers
from cascade.core.db import db_queries
from cascade.executor.construct_model import construct_model
from cascade.input_data.configuration.construct_bundle import (
    normalized_bundle_from_database,
    normalized_bundle_from_disk,
    bundle_to_observations
)
from cascade.input_data.db.asdr import asdr_as_fit_input
from cascade.input_data.db.locations import location_hierarchy
from cascade.executor.priors_from_draws import set_priors_from_parent_draws

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
    modified_data = modify_input_data(input_data, local_settings)
    model = construct_model(modified_data, local_settings)
    set_priors_from_parent_draws(model, input_data.draws)
    computed_fit = compute_location(modified_data.observations, local_settings)
    save_outputs(computed_fit, execution_context, local_settings)


def retrieve_data(execution_context, local_settings):
    data = SimpleNamespace()
    data_access = local_settings.data_access
    model_version_id = data_access.model_version_id

    data.locations = location_hierarchy(
        data_access.gbd_round_id, location_set_version_id=data_access.location_set_version_id)

    if local_settings.data_access.bundle_file:
        data.bundle = normalized_bundle_from_disk(local_settings.data_access.bundle_file)
    else:
        data.bundle = normalized_bundle_from_database(
            execution_context,
            model_version_id,
            bundle_id=local_settings.data_access.bundle_id,
            tier=local_settings.data_access.tier
        )

    ages_df = db_queries.get_age_metadata(
        age_group_set_id=local_settings.data_access.age_group_set_id,
        gbd_round_id=local_settings.data_access.gbd_round_id
    )

    # This comes in yearly from 1950 to 2018
    data.age_specific_death_rate = asdr_as_fit_input(
        local_settings.parent_location_id, local_settings.sex_id,
        local_settings.data_access.gbd_round_id, ages_df, with_hiv=local_settings.data_access.with_hiv)

    # These are the draws as output of the parent location.
    data.draws = None

    return data


def modify_input_data(input_data, local_settings):
    ev_settings = local_settings.settings
    # These are suitable for input to the fit.
    input_data.observations = bundle_to_observations(
        input_data.bundle,
        local_settings.parent_location_id,
        ev_settings.eta
    )
    # ev_settings.data_eta_by_integrand is a dummy in form.py.
    MATHLOG.info(f"Ignoring data_eta_by_integrand")
    return input_data


def compute_location(input_data, local_settings):
    return None


def save_outputs(computed_fit, execution_context, local_settings):
    return None
