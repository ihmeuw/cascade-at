from cascade.core import getLoggers
from cascade.input_data.db.asdr import load_asdr_to_t3
from cascade.input_data.db.bundle import freeze_bundle
from cascade.input_data.db.csmr import load_csmr_to_t3
from cascade.input_data.db.locations import location_hierarchy, get_descendants

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

    cause_id = data_access.add_csmr_cause
    model_version_id = data_access.model_version_id

    not_just_children_but_all_descendants = True
    locations = location_hierarchy(
        data_access.gbd_round_id, location_set_version_id=data_access.location_set_version_id)
    location_and_children = get_descendants(
        locations, parent_id, children_only=not_just_children_but_all_descendants, include_parent=True)
    freeze_bundle(execution_context, model_version_id, data_access.bundle_id)

    if cause_id is not None:
        MATHLOG.info(
            f"Cause {cause_id} selected as CSMR source, "
            "freezing it's data if it has not already been frozen."
        )
        load_csmr_to_t3(execution_context, data_access, location_and_children)
    load_asdr_to_t3(execution_context, data_access, location_and_children)
