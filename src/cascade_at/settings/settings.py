import json

from cascade_at.settings.settings_configuration import SettingsConfiguration
from cascade_at.core.db import db_tools
from cascade_at.core.errors import SettingsError
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def load_settings(settings_json):
    """
    Loads settings from a settings_json.

    Parameters:
        settings_json: (dict) dictionary of settings

    Usage:
    >>> from cascade_at.settings.base_case import BASE_CASE
    >>> settings = load_settings(BASE_CASE)

    Returns:
        cascade_at.settings.configuration.Configuration
    """
    settings = SettingsConfiguration(settings_json)
    errors = settings.validate_and_normalize()
    if errors:
        print(f"Configuration does not validate {errors}")
        raise SettingsError("Configuration does not validate", errors,
                            settings_json)
    return settings


def settings_json_from_model_version_id(model_version_id, conn_def):
    """
    Loads settings for a specific model version ID into a json.

    Parameters:
        model_version_id: (int) the model version ID
        conn_def: (str) the connection definition like 'dismod-at-dev'
    """
    df = db_tools.ezfuncs.query(
        f"""SELECT parameter_json FROM epi.model_version_at
            WHERE model_version_id = {model_version_id}""",
        conn_def=conn_def
    )
    return json.loads(df['parameter_json'][0])


def settings_from_model_version_id(model_version_id, conn_def):
    """
    Loads settings for a specific model version ID.

    Parameters:
        model_version_id: (int) the model version ID
        conn_def: (str) the connection definition like 'dismod-at-dev'

    Usage:
    >>> settings = settings_from_model_version_id(model_version_id=395837,
    >>>                                           conn_def='dismod-at-dev')
    """
    parameter_json = settings_json_from_model_version_id(model_version_id=model_version_id, conn_def=conn_def)
    settings = load_settings(parameter_json)
    return settings
