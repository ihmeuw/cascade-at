import numpy as np
from scipy.special import logit


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


RELABEL_INCIDENCE_MAP = {
    1: {
        'incidence': 'Sincidence',
        'Sincidence': 'Sincidence',
        'Tincidence': 'Tincidence'
    },
    2: {
        'incidence': 'Tincidence',
        'Sincidence': 'Sincidence',
        'Tincidence': 'Tincidence'
    },
    3: {
        'incidence': 'Sincidence',
        'Sincidence': 'Sincidence',
        'Tincidence': 'Tincidence'
    }
}
