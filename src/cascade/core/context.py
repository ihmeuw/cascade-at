from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

from cascade.core.parameters import ParameterProperty
from cascade.core.input_data import InputData


@contextmanager
def scratch_maker():
    """ Create a scratch directory."""
    try:
        # scratch_dir will be in cwd and named tmp<something>
        scratch_dir = Path(tempfile.mkdtemp(dir="."))
        yield scratch_dir
    finally:
        shutil.rmtree(scratch_dir)


class ExecutionContext:
    """
    This is a container for all information about the environment in which a
    model executes. This includes paths to data sources, information about
    cluster resources etc.
    """
    
    def __init__(self):
        self._dismodfile = None

    parameters = ParameterProperty()

    @property
    def dismodfile(self):
        return self._dismodfile

    def scratch_dir(self):
        return scratch_maker()


class _ModelParameters:
    cascade = ParameterProperty()
    node = ParameterProperty()


class ModelContext:
    """
    This is a container for all inputs, parametrization and data, necessary
    to run the model for a node in the hierarchy. It does not include any
    information about the computational environment in which the model is
    running, no file paths or cluster information.
    """

    def __init__(self):
        self.parameters = _ModelParameters()
        self.input_data = InputData()
