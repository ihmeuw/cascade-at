from .parameters import ParameterProperty
from .input_data import InputData


class ExecutionContext:
    """
    This is a container for all information about the environment in which a
    model executes. This includes paths to data sources, information about
    cluster resources etc.
    """

    parameters = ParameterProperty()


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
