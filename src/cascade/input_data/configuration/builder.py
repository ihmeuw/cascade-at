""" Functions for creating internal model representations of settings from EpiViz
"""
import numpy as np
from scipy.special import logit

from cascade.core.log import getLoggers
from cascade.input_data import InputDataError

CODELOG, MATHLOG = getLoggers(__name__)


def identity(x):
    return x


def squared(x):
    return np.power(x, 2)


def scale1000(x):
    return x * 1000


COVARIATE_TRANSFORMS = {0: identity, 1: np.log, 2: logit, 3: squared, 4: np.sqrt, 5: scale1000}
"""
These functions transform covariate data, as specified in EpiViz.
"""


class SettingsToModelError(InputDataError):
    """Error creating a model from the settings"""


def policies_from_settings(settings):
    return dict(settings.policies.items())
