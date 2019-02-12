from cascade.core import getLoggers
from cascade.input_data.db.asdr import load_asdr_to_t3
from cascade.input_data.db.bundle import freeze_bundle
from cascade.input_data.db.csmr import load_csmr_to_t3

CODELOG, MATHLOG = getLoggers(__name__)


def load_to_tier3(execution_context, bundle_id, model_version_id, parent_id, gbd_round_id):
    """Tier 2 is data from Central Computation processes and changes over time,
    so tier 3 stores the exact copy that was used for this run."""
    freeze_bundle(execution_context, bundle_id)

    if execution_context.parameters.add_csmr_cause is not None:
        MATHLOG.info(
            f"Cause {execution_context.parameters.add_csmr_cause} "
            "selected as CSMR source, freezing it's data if it has not already been frozen."
        )
        load_csmr_to_t3(execution_context, model_version_id)
    load_asdr_to_t3(execution_context, model_version_id, parent_id, gbd_round_id)
