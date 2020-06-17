from cascade_at.executor import cleanup
from cascade_at.executor import configure_inputs
from cascade_at.executor import dismod_db
from cascade_at.executor import upload
from cascade_at.executor import mulcov_statistics
from cascade_at.executor import predict_sample
from cascade_at.executor import run_dmdismod
from cascade_at.executor import sample_simulate


def _path_to_name(path: str) -> str:
    return path.split('.')[-1]


SCRIPT_LIST = [
    cleanup,
    configure_inputs,
    dismod_db,
    upload,
    mulcov_statistics,
    predict_sample,
    run_dmdismod,
    sample_simulate
]


ARG_DICT = {
    _path_to_name(script.__name__): script.ARG_LIST
    for script in SCRIPT_LIST
}
