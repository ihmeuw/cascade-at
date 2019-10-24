
from db_queries import get_envelope

from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def get_asdr(demographics, decomp_step,
             gbd_round_id, with_hiv=True):
    """
    Gets age-specific death rate for all
    demographic groups.

    :param demographics: (cascade_at.inputs.demographics.Demographics)
    :param decomp_step: (int)
    :param gbd_round_id: (int)
    :param with_hiv: (bool) pull HIV-added envelope?
    :return:
    """
    df = get_envelope(
        age_group_id=demographics.age_group_id,
        sex_id=demographics.sex_id,
        year_id=demographics.year_id,
        location_id=demographics.location_id,
        decomp_step=decomp_step,
        gbd_round_id=gbd_round_id,
        with_hiv=with_hiv
    )
    return df
