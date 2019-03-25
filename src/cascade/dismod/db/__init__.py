from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class DismodFileError(TypeError):
    """These are all Pandas dataframes that don't match what Dismod expects."""
