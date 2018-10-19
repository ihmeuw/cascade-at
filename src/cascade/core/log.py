import logging


def getLoggers(dotted_module_name):
    """
    Returns two loggers. The first is in the namespace that's passed-in.
    The second inserts ".math" as the second module, so ``cascade.core.log``
    becomes ``cascade.math.core.log``.

    Args:
        dotted_module_name (str): The name of the module, usually as a
            ``__name__`` variable at the top of the module after imports.

    Returns:
        logging.Logger: The logger to use for regular code logs.
        logging.Logger: The logger to use for messages about the statistics.
    """
    code_log = logging.getLogger(dotted_module_name)
    separated = code_log.name.split(".")
    math_log = logging.getLogger(".".join([separated[0], "math"] + separated[1:]))
    return code_log, math_log
