import logging


def get_loggers(dotted_module_name):
    log = logging.getLogger(dotted_module_name)
    return log
