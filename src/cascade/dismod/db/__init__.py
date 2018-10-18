from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class DismodFileError(Exception):
    pass
