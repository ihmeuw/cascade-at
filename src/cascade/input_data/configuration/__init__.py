from cascade.core.log import getLoggers
from cascade.input_data import InputDataError
CODELOG, MATHLOG = getLoggers(__name__)


class SettingsError(InputDataError):
    def __init__(self, message, form_errors=None, form_data=None):
        super().__init__(message)
        self.form_errors = form_errors if form_errors else list()
        self.form_data = form_data
