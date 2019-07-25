import json

from cascade.core.db import cursor
from cascade.core.log import getLoggers
from cascade.input_data.configuration import SettingsError
from cascade.input_data.configuration.form import Configuration

CODELOG, MATHLOG = getLoggers(__name__)


def load_settings(ec, meid=None, mvid=None, settings_file=None):
    CODELOG.debug(f"meid {meid} mvid {mvid} settings {settings_file}")
    if len([c for c in [meid, mvid, settings_file] if c is not None]) != 1:
        raise ValueError(
            "Must supply exactly one of mvid, meid or settings_file")
    if meid:
        raw_settings, found_mvid = load_raw_settings_meid(ec, meid)
    elif mvid:
        raw_settings, found_mvid = load_raw_settings_mvid(ec, mvid)
    elif settings_file:
        raw_settings, found_mvid = load_raw_settings_file(ec, settings_file)
    else:
        raise RuntimeError(f"Either meid, mvid, or file must be specified.")

    ec.parameters.model_version_id = found_mvid

    return json_settings_to_frozen_settings(raw_settings, found_mvid)


def load_raw_settings_meid(ec, modelable_entity_id):
    """
    Given a meid, get settings for the latest corresponding mvid.

    Args:
        modelable_entity_id (int,str): The MEID.
    Returns:
        dict: Settings as a JSON dictionary of dictionaries.
        int: Model version ID that is the latest one associated with this meid.
    """
    ec.parameters.modelable_entity_id = modelable_entity_id
    mvid = latest_model_version(ec)
    MATHLOG.info(
        f"No model version specified so using the latest version for "
        f"model {modelable_entity_id} which is {mvid}")
    raw_settings = settings_json_from_epiviz(ec, mvid)
    return raw_settings, mvid


def load_raw_settings_mvid(ec, mvid):
    """Given an mvid, get its settings.

    Args:
        mvid (int,str): Model version ID.
    Returns:
        dict: Settings as a JSON dictionary of dictionaries.
        int: Model version ID that was passed in.
    """
    raw_settings = settings_json_from_epiviz(ec, mvid)
    return raw_settings, mvid


def load_raw_settings_file(ec, settings_file):
    """Given a settings file, get the latest mvid for its meid.

    Args:
        settings_file (str): Model version ID.
    Returns:
        dict: Settings as a JSON dictionary of dictionaries.
        int: Model version ID, either found in file or latest from meid.
    """
    with open(str(settings_file), "r") as f:
        try:
            raw_settings = json.load(f)
        except json.decoder.JSONDecodeError as jde:
            MATHLOG.error(
                f"The format of the JSON in {settings_file} has an error. {jde}")
            raise
    if "model" in raw_settings and "modelable_entity_id" in raw_settings["model"]:
        ec.parameters.modelable_entity_id = raw_settings["model"]["modelable_entity_id"]
    else:
        raise SettingsError(
            f"The settings file should have a modelable_entity_id in it. "
            f"It would be under model and then modelable_entity_id.")
    if "model_version_id" in raw_settings["model"]:
        mvid = raw_settings["model"]["model_version_id"]
        MATHLOG.info(f"Using mvid {mvid} from the settings file.")
    else:
        mvid = latest_model_version(ec)
        MATHLOG.info(f"Using mvid {mvid} from latest model version.")
    return raw_settings, mvid


def json_settings_to_frozen_settings(raw_settings, mvid=None):
    """Converts a settings file in the form of a dict (from JSON usually)
    into a Configuration object. If that conversion fails, report errors
    with an exception.

    Args:
        raw_settings (dict): Dict of dicts, representing the JSON settings.
        mvid (int,optional): Model version ID to put into the settings.

    Returns:
        Configuration: Represents validated settings.
    """
    if "model_version_id" not in raw_settings["model"] or not raw_settings["model"]["model_version_id"]:
        raw_settings["model"]["model_version_id"] = mvid

    settings = Configuration(raw_settings)
    errors = settings.validate_and_normalize()
    if errors:
        print(f"Configuration does not validate {errors}")
        raise SettingsError("Configuration does not validate", errors,
                            raw_settings)
    return settings


def settings_json_from_epiviz(execution_context, model_version_id):
    query = """select parameter_json from at_model_parameter where model_version_id = %(model_version_id)s"""
    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        raw_data = c.fetchall()

    if len(raw_data) == 0:
        raise ValueError(f"No parameters for model version {model_version_id}")
    if len(raw_data) > 1:
        raise ValueError(f"Multiple parameter entries for model version {model_version_id}")

    config_data = json.loads(raw_data[0][0])

    # Fix bugs in epiviz
    # TODO: remove this once EPI-999 is resolved
    config_data = trim_config(config_data)
    if config_data is DO_REMOVE:
        config_data = {}

    return config_data


DO_REMOVE = object()


def trim_config(source):
    """ This function represents the approach to missing data which the viz
    team says it will enforce in the front end, though that hasn't happened
    yet.
    """
    trimmed = None
    remove = True
    if isinstance(source, dict):
        trimmed, remove = _trim_dict(source)
    elif isinstance(source, list):
        trimmed, remove = _trim_list(source)
    else:
        if source is not None and source != "":
            trimmed = source
            remove = False

    if remove:
        return DO_REMOVE
    else:
        return trimmed


def _trim_dict(source_dict):
    trimmed = {}
    remove = True
    for k, v in source_dict.items():
        # Removing keys prefixed by "__" because they represent garbage that
        # the GUI framework sticks in sometimes but isn't useful to us.
        if not k.startswith("__"):
            tv = trim_config(v)
            if tv is not DO_REMOVE:
                trimmed[k] = tv
                remove = False
    return trimmed, remove


def _trim_list(source_list):
    trimmed = []
    remove = True
    for v in source_list:
        tv = trim_config(v)
        if tv is not DO_REMOVE:
            trimmed.append(tv)
            remove = False
    return trimmed, remove


def latest_model_version(execution_context):
    model_id = execution_context.parameters.modelable_entity_id

    query = """
    select model_version_id from epi.model_version
    where modelable_entity_id = %(modelable_entity_id)s
    order by last_updated desc
    limit 1
    """

    with cursor(execution_context) as c:
        c.execute(query, args={"modelable_entity_id": model_id})
        result = c.fetchone()
        if result is not None:
            return result[0]
        else:
            raise RuntimeError(
                f"No model version for modelable entity id {model_id} in database.")
