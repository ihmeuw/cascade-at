from collections import namedtuple
from enum import Enum


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
    gamma = "meas_std"


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
"""Each integrand has a natural association with a particular weight because
it is a count of events with one of four denominators: constant, susceptibles,
with-condition, or the total population. For isntance, if you supply
mtspecific data, it will always use the weight called "total."
"""


INTEGRAND_COHORT_COST = dict(
    Sincidence=False,
    remission=False,
    mtexcess=False,
    mtother=False,
    susceptible=True,
    withC=True,
    prevalence=True,
    Tincidence=True,
    mtspecific=True,
    mtall=True,
    mtstandard=True,
    relrisk=False,
)
"""
If a value is True, then rate covariates on this integrand type are much
more costly than those that are False.
If a value is True, then for a data point of this integrand type,
if it has a rate covariate that is not equal to its reference value, then
the differential equation needs to be solved for each cohort in the
ODE grid that intersects or surrounds the age-time span for this point. This
computation is performed each time the model variables change.
"""


DismodIO = namedtuple("CommandIO", "input output")
"""Each Dismod-AT command reads and writes subsets of tables.
These are those tables."""


COMMAND_IO = dict(
    init=DismodIO(
        ["age", "avgint", "covariate", "data", "density", "integrand",
         "mulcov", "node", 'nslist_pair', "nslist", "option",
         "prior", "rate", "smooth_grid", "smooth", "time",
         "weight_grid", "weight"],
        ["age_avg", "var", "data_subset", "start_var", "scale_var", "truth_var"]),
    depend=DismodIO([], ["depend_var", "age_avg"]),
    set=DismodIO(
        ["sample", "prior_mean", "fit_var", "start_var", "scale_var"],
        ["start_var", "scale_var", "truth_var"]),
    fit=DismodIO(["data_sim"], ["fit_var", "fit_data_subset", "age_avg", "truth_var"]),
    simulate=DismodIO(["truth_var"], ["data_sim", "age_avg"]),
    sample=DismodIO(["fit_var", "data_sim", "prior_sim"],
                    ["sample", "age_avg"]),
    predict=DismodIO(["sample", "fit", "truth"], ["predict", "age_avg"]),
)
"""Given a command, what tables does it read and write?
This is used to decide which tables must be refreshed after Dismod-AT
executes a command. We do this because refreshing the data table can
take tens of minutes, not seconds.
"""
