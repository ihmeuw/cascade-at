import elmo

from cascade_at.inputs.utilities import ids
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def get_crosswalk_version(crosswalk_version_id, exclude_outliers, conn_def):
    """
    Downloads crosswalk version data specified by the crosswalk version ID.

    :param crosswalk_version_id: (int)
    :param exclude_outliers: (bool) whether to exclude outliers
    :param conn_def: (str) connection definition
    :return: pd.DataFrame
    """
    LOG.info(f"Getting crosswalk version for {crosswalk_version_id}.")
    df = elmo.get_crosswalk_version(crosswalk_version_id=crosswalk_version_id)
    if exclude_outliers:
        df = df.loc[df.is_outlier != 1].copy()

    sex_ids = ids.get_sex_ids()
    measure_ids = ids.get_measure_ids(conn_def=conn_def)

    df = df.merge(sex_ids, on='sex')
    df = df.merge(measure_ids, on='measure')

    df = df.loc[~df.input_type.isin(['parent', 'group_review'])].copy()

    return df
