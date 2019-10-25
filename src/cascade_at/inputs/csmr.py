from db_queries import get_outputs
import gbd.constants as gbd

from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def get_csmr(process_version_id,
             cause_id,
             demographics,
             decomp_step,
             gbd_round_id):
    """
    Get cause-specific mortality rate
    for demographic groups from a specific
    CodCorrect output version.

    :param process_version_id: (int)
    :param cause_id: (int)
    :param demographics (cascade_at.inputs.demographics.Demographics)
    :param decomp_step: (str)
    :param gbd_round_id: (int)
    :return:
    """
    LOG.info(f"Getting CSMR from process version ID {process_version_id}")
    df = get_outputs(
        topic='cause',
        cause_id=cause_id,
        metric_id=gbd.metrics.RATE,
        measure_id=gbd.measures.DEATH,
        year_id=demographics.year_id,
        location_id=demographics.location_id,
        sex_id=demographics.sex_id,
        age_group_id=demographics.age_group_id,
        gbd_round_id=gbd_round_id,
        decomp_step='step4',
        process_version_id=14469
    )
    df = df.rename(columns={
        "val": "meas_value",
        "year_id": "time_lower"
    })
    df["time_upper"] = df["time_lower"] + 1
    return df

