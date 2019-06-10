from math import nan

from cascade.core import getLoggers
from cascade.core.subprocess_utils import run_with_logging
from cascade.dismod.constants import COMMAND_IO
from cascade.dismod.process_behavior import check_command
from cascade.model.object_wrapper import ObjectWrapper

CODELOG, MATHLOG = getLoggers(__name__)


def prepare_init(dismod_objects, locations, parent_location, model, data):
    """These steps prepare a model for running Dismod-AT init."""
    dismod_objects.locations = locations
    dismod_objects.parent_location_id = parent_location
    dismod_objects.model = model
    if data is not None:
        # The data has to be in there before init in order to build
        # the data subset table.
        data = _point_age_time_to_interval(data)
        data = amend_data_input(data)
        dismod_objects.data = data


def prepare_fit(dismod_objects, initial_guess=None, scale=None):
    """Prepare a model for calling a fit command."""
    if scale is not None:
        dismod_objects.scale_var = scale
    # else Dismod uses the initial guess as the scale.
    if initial_guess is not None:
        dismod_objects.start_var = initial_guess
    # else Dismod-AT uses the distribution means as the start_var.


def prepare_predict(dismod_objects, avgint, var):
    avgint = _point_age_time_to_interval(avgint)
    dismod_objects.avgint = avgint
    dismod_objects.truth_var = var


def run_dismod(dismod_objects, command):
    """Pushes tables to the db file, runs Dismod-AT, and refreshes
    tables written. This flushes the in-memory objects before
    running Dismod.

    Args:
        dismod_objects (ObjectWrapper): Handle to Dismod tables as objects.
        command (List[str]|str): Command to run as a list of strings
            or a single string without spaces.
    Returns:
        (str, str): Stdout and stderr as strings, not bytes.
    """
    assert isinstance(dismod_objects, ObjectWrapper)
    if isinstance(command, str):
        command = [command]
    dismod_objects.flush()
    CODELOG.debug(f"Running Dismod-AT {command}")
    with dismod_objects.close_db_while_running():
        str_command = [str(c) for c in command]
        return_code, stdout, stderr = run_with_logging(
            ["dmdismod", str(dismod_objects.db_filename)] + str_command)

    log = dismod_objects.log
    check_command(str_command[0], log, return_code, stdout, stderr)
    if command[0] in COMMAND_IO:
        dismod_objects.refresh(COMMAND_IO[command[0]].output)
    return stdout, stderr


def _point_age_time_to_interval(data):
    if data is None:
        return
    for at in ["age", "time"]:  # Convert from point ages and times.
        for lu in ["lower", "upper"]:
            if f"{at}_{lu}" not in data.columns and at in data.columns:
                data = data.assign(**{f"{at}_{lu}": data[at]})
    return data.drop(columns={"age", "time"} & set(data.columns))


def amend_data_input(data):
    """If the data comes in without optional entries, add them.
    This doesn't translate to internal IDs for Dismod-AT. It rectifies
    the input, and this is how it should be saved or passed to another tool.
    """
    if data is None:
        return

    data = _point_age_time_to_interval(data)

    if "name" not in data.columns:
        data = data.assign(name=data.index.astype(str))
    else:
        null_names = data[data.name.isnull()]
        if not null_names.empty:
            raise ValueError(f"There are some data values that lack data names. {null_names}")

    if "hold_out" not in data.columns:
        data = data.assign(hold_out=0)
    for additional in ["nu", "eta"]:
        if additional not in data.columns:
            data = data.assign(**{additional: nan})
    return data
