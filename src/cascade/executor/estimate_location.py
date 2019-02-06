from types import SimpleNamespace

from cascade.core import getLoggers
from cascade.input_data.configuration.construct_bundle import (
    normalized_bundle_from_database,
    normalized_bundle_from_disk,
    bundle_to_observations
)

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
    model_version_id = local_settings.model.model_version_id

    if execution_context.parameters.bundle_file:
        data.bundle = normalized_bundle_from_disk(execution_context.parameters.bundle_file)
    else:
        bundle_id = local_settings.model.bundle_id
        data.bundle = normalized_bundle_from_database(
            execution_context,
            model_version_id,
            bundle_id=bundle_id,
            tier=execution_context.parameters.tier
        )

    return data


def compute_location(input_data, execution_context, local_settings):

    parent_location_id = local_settings.model.parent_location_id
    global_data_eta = local_settings.eta
    observations = bundle_to_observations(input_data.bundle, parent_location_id, global_data_eta)

    return observations


def save_outputs(computed_fit, execution_context, local_settings):
    return None
