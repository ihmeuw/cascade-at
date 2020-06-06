import logging


LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def get_loggers(dotted_module_name):
    log = logging.getLogger(dotted_module_name)
    return log
