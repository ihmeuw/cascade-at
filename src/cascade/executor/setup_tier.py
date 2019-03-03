from cascade.core import getLoggers
from cascade.input_data.configuration.construct_mortality import location_and_children_from_settings
from cascade.input_data.db.asdr import load_asdr_to_t3
from cascade.input_data.db.bundle import freeze_bundle
from cascade.input_data.db.csmr import load_csmr_to_t3

CODELOG, MATHLOG = getLoggers(__name__)


def setup_tier_data(execution_context, data_access, parent_id):
    """
    The Tiers refer to places IHME data is stored. Tier 2 data can change
    over time. Tier 3 is a copy of the data the is immutable in order to
    record exactly what was input for this run.

    Args:
        execution_context:
        data_access: Parameter block with constants for getting data
        parent_id (int): Parent location ID.
    """
    if data_access.tier != 3:
        return

    freeze_bundle(execution_context, data_access.model_version_id, data_access.bundle_id)

    location_and_children = location_and_children_from_settings(data_access, parent_id)
    if data_access.add_csmr_cause is not None:
        MATHLOG.info(
            f"Cause {data_access.add_csmr_cause} selected as CSMR source, "
            "freezing it's data if it has not already been frozen."
        )
        load_csmr_to_t3(execution_context, data_access, location_and_children)
    load_asdr_to_t3(execution_context, data_access, location_and_children)
