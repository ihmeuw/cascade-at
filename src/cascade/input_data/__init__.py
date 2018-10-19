from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class InputDataError(Exception):
    """These are errors that result from faults in the input data."""
