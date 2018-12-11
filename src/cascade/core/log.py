from functools import wraps
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


def logged(logger):
    """
    Decorator to log calls to a function::

        @logged(MATHLOG)
        def calculate_things(count, items): pass

    This will send all calls to calculate things to the MATHLOG logger.
    If you want to use it when calling a function outside this library,
    it's a little funny::

        logged(CODELOG)(db_queries.get_covariates)(covariate_id, age_group_id)

    That will work, though.

    Args:
        logger (logging.Logger): Something like a standard Python logger.

    Returns:
        A function because it's a decorator.
    """
    def logged_wrap(callable):
        @wraps(callable)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {getattr(callable, '__name__', 'func')}({args}, {kwargs})")
            return callable(*args, **kwargs)
        return wrapper
    return logged_wrap
