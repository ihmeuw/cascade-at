from cascade.core.context import ExecutionContext
from uuid import uuid4

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def make_execution_context(**parameters):
    defaults = {"database": "dismod-at-dev", "bundle_database": "epi"}
    defaults.update(parameters)
    context = ExecutionContext()
    context.parameters = defaults
    context.parameters.run_id = uuid4()
    return context
