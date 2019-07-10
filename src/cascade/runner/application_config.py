from functools import lru_cache

import pkg_resources
import toml


@lru_cache(maxsize=1)
def application_config():
    """
    Read application configuration, first from a local configuration file,
    then from a configuration file that's specific to this installation.
    We split the installations so that paths that are specific to the
    Institute are kept local.

    The data files are in TOML because it's a predictable form of ini file.

    Returns:
        Dict[str,str]: A dictionary of settings.
    """
    base_data = pkg_resources.resource_string("cascade.executor", "data/config.toml")
    parameters = toml.loads(base_data.decode())
    try:
        if pkg_resources.resource_exists("cascade.local_config", "config.toml"):
            raw_data = pkg_resources.resource_string("cascade.local_config", "config.toml")
            parameters.update(toml.loads(raw_data.decode()))
    except ModuleNotFoundError:
        pass  # This won't be installed unless it's an infrastructure install.
    parameters.update(parameters)
    return parameters
