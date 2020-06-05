from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class InputDataError(Exception):
    """These are errors that result from faults in the input data."""


class SettingsError(InputDataError):
    def __init__(self, message, form_errors=None, form_data=None):
        super().__init__(message)
        self.form_errors = form_errors if form_errors else list()
        self.form_data = form_data


class CascadeError(Exception):
    """Cascade base for exceptions."""


class DismodFileError(TypeError):
    """These are all Pandas data frames that don't match what Dismod expects."""
