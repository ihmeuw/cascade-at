from enum import Enum
import pandas as pd
import numpy as np


def enum_to_dataframe(enum_name):
    """Given an enum, return a dataframe with two columns, name and value."""
    return pd.DataFrame.from_records(
        np.array(
            [(measure, enum_value.value) for (measure, enum_value) in enum_name.__members__.items()],
            dtype=np.dtype([("name", object), ("value", int)]),
        )
    )


class DensityEnum(Enum):
    """The distributions supported by Dismod-AT. They always have these ids."""
    uniform = 0
    "Uniform Distribution"
    gaussian = 1
    "Gaussian Distribution"
    laplace = 2
    "Laplace Distribution"
    students = 3
    "Students-t Distribution"
    log_gaussian = 4
    "Log-Gaussian Distribution"
    log_laplace = 5
    "Log-Laplace Distribution"
    log_students = 6
    "Log-Students-t Distribution"


class RateEnum(Enum):
    """These are the five underlying rates.
    """
    pini = 0
    """Initial prevalence of the condition at birth, as a fraction of one."""
    iota = 1
    """Incidence rate for leaving susceptible to become diseased."""
    rho = 2
    """Remission from disease to susceptible."""
    chi = 3
    """Excess mortality rate."""
    omega = 4
    """Other-cause mortality rate."""


class MulCovEnum(Enum):
    """These are the mulcov kinds listed in the mulcov table."""
    alpha = "rate_value"
    beta = "meas_value"
    gamma = "meas_noise"


class IntegrandEnum(Enum):
    """These are all of the integrands Dismod-AT supports, and they will
    have exactly these IDs when serialized."""
    Sincidence = 0
    """Susceptible incidence, where the denominator is the number of susceptibles.
    Corresponds to iota."""
    remission = 1
    """Remission rate, corresponds to rho."""
    mtexcess = 2
    """Excess mortality rate, corresponds to chi."""
    mtother = 3
    """Other-cause mortality, corresponds to omega."""
    mtwith = 4
    """Mortality rate for those with condition."""
    susceptible = 5
    """Fraction of susceptibles out of total population."""
    withC = 6
    """Fraction of population with the disease. Total pop is the denominator."""
    prevalence = 7
    """Fraction of those alive with the disease, so S+C is denominator."""
    Tincidence = 8
    """Total-incidence, where denominator is susceptibles and with-condition."""
    mtspecific = 9
    """Cause-specific mortality rate, so mx_c."""
    mtall = 10
    """All-cause mortality rate, mx."""
    mtstandard = 11
    """Standardized mortality ratio."""
    relrisk = 12
    """Relative risk."""
    incidence = -99
    """This integrand should never be used, but we need it when we are converting
       from the epi database measures initially"""


class WeightEnum(Enum):
    """Dismod-AT allows arbitrary weights, which are functions of space
    and time, defined by bilinear interpolations on grids. These weights
    are used to average rates over age and time intervals. Given this
    problem, there are three kinds of weights that are relevant."""
    constant = 0
    """This weight is constant everywhere at 1. This is the no-weight weight."""
    susceptible = 1
    """For measures that are integrals over population without the condition."""
    with_condition = 2
    """For measures that are integrals over those with the disease."""
    total = 3
    """For measures where the denominator is the whole population."""


class PriorKindEnum(Enum):
    """The three kinds of priors."""
    value = 0
    dage = 1
    dtime = 2


INTEGRAND_TO_WEIGHT = dict(
    Sincidence=WeightEnum.susceptible,
    remission=WeightEnum.with_condition,
    mtexcess=WeightEnum.with_condition,
    mtother=WeightEnum.total,
    susceptible=WeightEnum.constant,
    withC=WeightEnum.constant,
    mtwith=WeightEnum.with_condition,
    prevalence=WeightEnum.total,
    Tincidence=WeightEnum.total,
    mtspecific=WeightEnum.total,
    mtall=WeightEnum.total,
    mtstandard=WeightEnum.constant,
    relrisk=WeightEnum.constant,
)


RateToIntegrand = {
    "iota": "Sincidence",
    "rho": "remission",
    "chi": "mtexcess",
    "omega": "mtother",
    "pini": "prevalence"
}

IntegrandToRate = {
    v: k for k, v in RateToIntegrand.items()
}

"""Each integrand has a natural association with a particular weight because
it is a count of events with one of four denominators: constant, susceptibles,
with-condition, or the total population. For instance, if you supply
mtspecific data, it will always use the weight called "total."
"""
