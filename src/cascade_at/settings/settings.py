import json

from cascade_at.settings.base_case import BASE_CASE
from cascade_at.settings.configuration import Configuration
from cascade_at.core.errors import SettingsError
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


def load_settings(settings_json):
    import pdb; pdb.set_trace()
    settings = Configuration(settings_json)
    errors = settings.validate_and_normalize()
    if errors:
        print(f"Configuration does not validate {errors}")
        raise SettingsError("Configuration does not validate", errors,
                            settings_json)
    return settings

