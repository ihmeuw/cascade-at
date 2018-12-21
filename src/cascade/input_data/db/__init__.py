from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

AGE_GROUP_SET_ID = 12

METRIC_IDS = {"per_capita_rate": 3}

MEASURE_IDS = {"deaths": 1}


class GBDDataError(Exception):
    """This error represents an unrecoverable problem with the data in the
    GBD databases. It is likely not something that the modelers can resolve
    by changing their model and should probably be reported to Central Comp.
    """
