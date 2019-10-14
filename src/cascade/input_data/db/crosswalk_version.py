"""This module provides tools for working directly with bundle data in the external databases. Code which wants to
manipulate the bundles directly in the database should live here but bundle code which does not need to access the
databases directly should live outside the db package and use the functions here to retrieve the data in normalized
form.
"""

import elmo

from cascade.core.db import cursor
from cascade.input_data import InputDataError
from cascade.core.log import getLoggers
from cascade.input_data.db.id_maps import map_variables_to_id

CODELOG, MATHLOG = getLoggers(__name__)


def _get_crosswalk_version_id(execution_context, model_version_id):
    """Gets the crosswalk version id associated with the current model_version_id.
    """
    query = f"""SELECT crosswalk_version
                FROM epi.model_version_at
                WHERE model_version_id = {model_version_id}"""

    with cursor(execution_context) as c:
        CODELOG.debug(f"Looking up crosswalk_version_id for model_version_id {model_version_id}")
        c.execute(query, args={"model_version_id": model_version_id})
        crosswalk_version_ids = list(c)

        if not crosswalk_version_ids:
            raise InputDataError(f"No crosswalk_version_id associated with model_version_id {model_version_id}")

        if len(crosswalk_version_ids) > 1:
            raise InputDataError(f"Multiple crosswalk_version_ids associated with model_version_id {model_version_id}")

        return crosswalk_version_ids[0][0]


def _get_crosswalk_version(crosswalk_version_id, exclude_outliers=True):
    """
    Downloads crosswalk version data specified by the crosswalk_version_id.

    Returns:
        Crosswalk Version data, retrieved from a crosswalk version.
    """
    crosswalk_version = elmo.get_crosswalk_version(crosswalk_version_id=crosswalk_version_id)
    if exclude_outliers:
        crosswalk_version = crosswalk_version.loc[crosswalk_version.is_outlier != 1].copy()
    crosswalk_version = map_variables_to_id(crosswalk_version, variables=['sex', 'measure'])
    crosswalk_version = crosswalk_version.loc[~crosswalk_version.input_type.isin(['parent', 'group_review'])].copy()

    MATHLOG.debug(f"Downloaded {len(crosswalk_version)} lines of crosswalk_version_id {crosswalk_version_id}")
    if exclude_outliers:
        # The modelers input the group_review flag as group_review=0 but then elmo transforms it to
        # input_type = 'group_review' which is what we actually filter on above, along with getting
        # rid of 'parent' input types.
        MATHLOG.debug("This excludes rows marked as outliers as well as those marked as group_review and parent")
    else:
        MATHLOG.debug("This excludes rows marked as group_review and parent")

    return crosswalk_version



