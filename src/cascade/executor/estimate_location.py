from types import SimpleNamespace

from cascade.core import getLoggers
from cascade.core.db import db_queries
from cascade.input_data.configuration.construct_bundle import (
    normalized_bundle_from_database,
    normalized_bundle_from_disk,
    bundle_to_observations
)
from cascade.input_data.db.asdr import asdr_as_fit_input
from cascade.input_data.db.locations import location_hierarchy

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
    computed_fit = compute_location(input_data, execution_context, local_settings)
    save_outputs(computed_fit, execution_context, local_settings)


def retrieve_data(execution_context, local_settings):
    data = SimpleNamespace()
    model_version_id = local_settings.data_access.model_version_id

    data.locations = location_hierarchy(execution_context)

    if local_settings.data_access.bundle_file:
        data.bundle = normalized_bundle_from_disk(local_settings.data_access.bundle_file)
    else:
        bundle_id = local_settings.data_access.model.bundle_id
        data.bundle = normalized_bundle_from_database(
            execution_context,
            model_version_id,
            bundle_id=bundle_id,
            tier=local_settings.data_access.tier
        )

    ages_df = db_queries.get_age_metadata(
        age_group_set_id=local_settings.policies.age_group_set_id,
        gbd_round_id=local_settings.ihme_parameters.gbd_round_id
    )

    # This comes in yearly from 1950 to 2018
    data.age_specific_death_rate = asdr_as_fit_input(
        local_settings.parent_location_id, local_settings.sex_id,
        local_settings.data_access.gbd_round_id, ages_df, with_hiv=local_settings.data_access.with_hiv)

    return data


def compute_location(input_data, execution_context, local_settings):

    parent_location_id = local_settings.parent_location_id
    global_data_eta = local_settings.settings.eta
    observations = bundle_to_observations(input_data.bundle, parent_location_id, global_data_eta)

    return observations


def save_outputs(computed_fit, execution_context, local_settings):
    return None
