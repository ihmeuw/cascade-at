from cascade.core.context import ExecutionContext

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def make_execution_context(**parameters):
    defaults = {"database": "dismod-at-dev", "bundle_database": "epi"}
    defaults.update(parameters)
    context = ExecutionContext()
    context.parameters = defaults
    return context
