from cascade.core.context import ExecutionContext


def make_execution_context(**parameters):
    defaults = {"database": "dismod-at-dev", "bundle_database": "epi"}
    defaults.update(parameters)
    context = ExecutionContext()
    context.parameters = defaults
    return context
