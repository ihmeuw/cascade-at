from cascade_at.executor import configure_inputs


def _path_to_name(path: str) -> str:
    return path.split('.')[-1]


SCRIPT_LIST = [
    configure_inputs
]


ARG_DICT = {
    _path_to_name(script.__name__): script.ARG_LIST
    for script in SCRIPT_LIST
}
