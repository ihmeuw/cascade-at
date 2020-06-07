import json
from typing import Dict, Any

from cascade_at.settings.settings_config import SettingsConfig
from cascade_at.core.db import db_tools
from cascade_at.core.errors import SettingsError
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def load_settings(settings_json: Dict[str, Any]) -> SettingsConfig:
    """
    Loads settings from a settings_json.

    Parameters
    ==========
    settings_json
        dictionary of settings

    Examples
    ========
    >>> from cascade_at.settings.base_case import BASE_CASE
    >>> settings = load_settings(BASE_CASE)
    """
    settings = SettingsConfig(settings_json)
    errors = settings.validate_and_normalize()
    if errors:
        print(f"Configuration does not validate {errors}")
        raise SettingsError("Configuration does not validate", errors,
                            settings_json)
    return settings


def settings_json_from_model_version_id(model_version_id: int, conn_def: str) -> Dict[str, any]:
    """
    Loads settings for a specific model version ID into a json.

    Parameters
    ==========
    model_version_id
        the model version ID
    conn_def
        the connection definition like 'dismod-at-dev'
    """
    df = db_tools.ezfuncs.query(
        f"""SELECT parameter_json FROM epi.model_version_at
            WHERE model_version_id = {model_version_id}""",
        conn_def=conn_def
    )
    return json.loads(df['parameter_json'][0])


def settings_from_model_version_id(model_version_id: int, conn_def: str) -> SettingsConfig:
    """
    Loads settings for a specific model version ID.

    Parameters
    ==========
    model_version_id
        the model version ID
    conn_def
        the connection definition like 'dismod-at-dev'

    Examples
    ========
    >>> settings = settings_from_model_version_id(model_version_id=395837,
    >>>                                           conn_def='dismod-at-dev')
    """
    parameter_json = settings_json_from_model_version_id(
        model_version_id=model_version_id, conn_def=conn_def
    )
    settings = load_settings(parameter_json)
    return settings
