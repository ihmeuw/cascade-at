from configparser import ConfigParser
from functools import lru_cache
from os import linesep

from pkg_resources import resource_string, iter_entry_points

from cascade_at.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


@lru_cache(maxsize=1)
def application_config():
    """Returns a utilities dictionary.
    If something passes in an object of type ConfigParser,
    then we use that.

    Args:
        alternate_configparser (ConfigParser.SectionProxy):
            If this is passed in, then use this instead of
            the internal config parser.

    Returns:
        ConfigParser.SectionProxy: This is a mapping type.
    """
    parser = ConfigParser()
    bytes_data = resource_string("cascade_at.executor", "data/config.cfg")
    parser.read_string(bytes_data.decode())
    config_sources = list()
    for entry_point in iter_entry_points("ihmeuw.config", "cascade_at"):
        CODELOG.debug(f"Found utilities in distribution {entry_point.dist}")
        config_sources.append(entry_point.dist)
        parser.read_dict(entry_point.load()())
    if len(config_sources) > 1:
        MATHLOG.info(
            f"More than one utilities, loaded in the order {linesep}"
            f"{linesep.join(str(cs) for cs in config_sources)}"
        )
    return parser
