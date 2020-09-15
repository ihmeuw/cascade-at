from configparser import ConfigParser
from os import linesep

from pkg_resources import iter_entry_points


from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def application_config():
    """Returns a configuration dictionary based on the configuration
    that is installed into the environment.

    Returns
    -------
    ConfigParser.SectionProxy
        This is a mapping type.
    """

    parser = ConfigParser()
    config_sources = list()

    for entry_point in iter_entry_points("config"):
        LOG.debug(f"Found configuration in distribution {entry_point.dist}")
        config_sources.append(entry_point.dist)
        parser.read_dict(entry_point.load()())
    if len(config_sources) > 1:
        LOG.info(
            f"More than one configuration, loaded in the order {linesep}"
            f"{linesep.join(str(cs) for cs in config_sources)}"
        )
    return parser
