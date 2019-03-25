from pkg_resources import get_distribution, DistributionNotFound

from cascade.core.log import getLoggers

try:
    __version__ = get_distribution("cascade").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unavailable"


class CascadeError(Exception):
    """Cascade base for exceptions."""


__all__ = ["getLoggers", "__version__", "CascadeError"]
