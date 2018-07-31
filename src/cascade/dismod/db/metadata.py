"""
This describes the tables in the sqlite file that Dismod reads.
"""
import enum
import logging

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Enum, ForeignKey


LOGGER = logging.getLogger(__name__)

Base = declarative_base()


class Age(Base):
    __tablename__ = "age"

    age_id = Column(Integer(), primary_key=True)
    age = Column(Float(), unique=True, nullable=False)


class Time(Base):
    __tablename__ = "time"

    time_id = Column(Integer(), primary_key=True)
    time = Column(Float(), unique=True, nullable=False)


class IntegrandEnum(enum.Enum):
    Sincidence = 0
    remission = 1
    mtexcess = 2
    mtother = 3
    mtwith = 4
    susceptible = 5
    withC = 6
    prevalence = 7
    Tincidence = 8
    mtspecific = 9
    mtall = 10
    mtstandard = 11
    relrisk = 12


class Integrand(Base):
    """These are the names of the integrands, taken from IntegrandEnum"""

    __tablename__ = "integrand"

    integrand_id = Column(Integer(), primary_key=True)
    integrand_name = Column(Enum(IntegrandEnum), unique=True, nullable=False)
    """
    Each integrand may appear only once. Unused integrands need not be added.
    """
    minimum_meas_cv = Column(Float())


class DensityEnum(enum.Enum):
    uniform = 0
    gaussian = 1
    laplace = 2
    students = 3
    log_gaussian = 4
    log_laplace = 5
    log_students = 6


class Density(Base):
    __tablename__ = "density"

    density_id = Column(Integer(), primary_key=True)
    density_name = Column(Enum(DensityEnum), unique=True, nullable=False)


class Covariate(Base):
    """The names of covariates and some reference values"""

    __tablename__ = "covariate"

    covariate_id = Column(Integer(), primary_key=True)
    covariate_name = Column(String(), nullable=False, unique=True)
    reference = Column(Float(), nullable=False)
    """The value of the covariate that corresponds to no adjustment"""
    max_difference = Column(Float(), nullable=True)
    """
    Maximum absolute difference from the reference value. Must be greater
    than or equal to zero. Null is interpreted as positive infinity, which
    means no exclusion is done for this covariate.
    """


class Node(Base):
    """
    These are locations, and they form a hierarchy, specified through parent.
    """

    __tablename__ = "node"

    node_id = Column(Integer(), primary_key=True)
    node_name = Column(String(), nullable=False, unique=True)
    parent = Column(None, ForeignKey("node.node_id"), nullable=True)  # Parent is an id in _this_ table.


class Prior(Base):
    __tablename__ = "prior"

    prior_id = Column(Integer(), primary_key=True)
    prior_name = Column(String(), unique=True)
    density_id = Column(None, ForeignKey("density.density_id"))
    lower = Column(Float(), nullable=True)
    upper = Column(Float(), nullable=True)
    mean = Column(Float(), nullable=False)
    std = Column(Float(), nullable=False)
    eta = Column(Float(), nullable=True)
    nu = Column(Float(), nullable=True)


class Weight(Base):
    __tablename__ = "weight"

    weight_id = Column(Integer(), primary_key=True)
    weight_name = Column(String(), unique=True)
    n_age = Column(Integer())
    """The number of age values in the smoothing grid. Greater than zero"""
    n_time = Column(Integer())
    """The number of time values in the smoothing grid. Greater than zero"""


class WeightGrid(Base):
    __tablename__ = "weight_grid"

    weight_grid_id = Column(Integer(), primary_key=True)
    weight_id = Column(None, ForeignKey("weight.weight_id"), nullable=False)
    age_id = Column(None, ForeignKey("age.age_id"), nullable=False)
    time_id = Column(None, ForeignKey("time.time_id"), nullable=False)
    weight = Column(Float(), nullable=False)
    """This is the weight for this age, time, and weight id."""


class Smooth(Base):
    __tablename__ = "smooth"

    smooth_id = Column(Integer(), primary_key=True, nullable=False)
    smooth_name = Column(String(), unique=True, nullable=False)
    n_age = Column(Integer())
    """The number of age values in the smoothing grid. Greater than zero"""

    n_time = Column(Integer(), nullable=False)
    """The number of time values in the smoothing grid. Greater than zero"""

    mulstd_value_prior_id = Column(None, ForeignKey("prior.prior_id"), nullable=True)
    """The prior_id for the variable that multiplies the value_prior_id
    standard deviations for this smooth_id"""

    mulstd_dage_prior_id = Column(None, ForeignKey("prior.prior_id"), nullable=True)
    """The prior_id for the variable that multiplies the age_prior_id
    standard deviations for this smooth_id"""

    mulstd_dtime_prior_id = Column(None, ForeignKey("prior.prior_id"), nullable=True)
    """The prior_id for the variable that multiplies the dtime_prior_id
    standard deviations for this smooth_id"""


class SmoothGrid(Base):
    __tablename__ = "smooth_grid"

    smooth_grid_id = Column(Integer(), primary_key=True)
    smooth_id = Column(Integer(), unique=True, nullable=True)
    age_id = Column(None, ForeignKey("age.age_id"), nullable=False)
    time_id = Column(None, ForeignKey("time.time_id"), nullable=False)
    value_prior_id = Column(Integer(), nullable=True)
    """A prior_id. If null, const_value must not be null."""

    dage_prior_id = Column(None, ForeignKey("prior.prior_id"), nullable=True)
    """A prior_id. If null, const_value must not be null."""

    dtime_prior_id = Column(None, ForeignKey("prior.prior_id"), nullable=True)
    """A prior_id. If null, const_value must not be null."""

    const_value = Column(Float(), nullable=True)
    """If non-null, specifies the value of the function at the corresponding
    age and time. It sets lower, upper, and mean to this value and density
    to uniform. If null, value_prior_id must not be null."""


class NSList(Base):
    __tablename__ = "nslist"

    nslist_id = Column(Integer(), primary_key=True)
    nslist_name = Column(String(), unique=True, nullable=False)


class NSListPair(Base):
    """Associates a node with a smoothing"""

    __tablename__ = "nslist_pair"

    nslist_pair_id = Column(Integer(), primary_key=True)
    nslist_id = Column(None, ForeignKey("nslist.nslist_id"), nullable=False)
    node_id = Column(None, ForeignKey("node.node_id"), nullable=False)
    smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=False)


class RateName(enum.Enum):
    pini = 0
    iota = 1
    rho = 2
    chi = 3
    omega = 4


class Rate(Base):
    __tablename__ = "rate"

    rate_id = Column(Integer(), primary_key=True)
    rate_name = Column(Enum(RateName), nullable=False)
    parent_smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=True)
    """If null, then parent rate is always zero and no model variables are
    allocated for it"""

    child_smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=True)
    child_nslist_id = Column(None, ForeignKey("nslist.nslist_id"), nullable=True)
    """If not null, identifies a list of node and smoothing pairs. The
    node_id for each of the children must appear in the list. The corresponding
    smoothing is used for that child and the rate corresponding to this
    row of the rate table."""


class MulCovEnum(enum.Enum):
    rate_value = 0
    meas_value = 1
    meas_std = 2


class MulCov(Base):
    """Covariate multipliers and their priors.

    If this is empty, there are no covariate multipliers in the model.
    """

    __tablename__ = "mulcov"

    mulcov_id = Column(Integer(), primary_key=True)
    mulcov_type = Column(Enum(MulCovEnum), nullable=False)
    rate_id = Column(None, ForeignKey("rate.rate_id"), nullable=True)
    """Determines the rate that this covariate and multiplier affects.
    If mulcov_type is of type meas_value or meas_std, this must be null."""
    integrand_id = Column(None, ForeignKey("integrand.integrand_id"), nullable=True)
    covariate_id = Column(None, ForeignKey("covariate.covariate_id"), nullable=False)
    smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=True)
    """If this is null, the covariate multiplier is always zero and no
    model_variables are allocated for it."""


class AvgInt(Base):
    __tablename__ = "avgint"

    avgint_id = Column(Integer(), primary_key=True)
    integrand_id = Column(None, ForeignKey("integrand.integrand_id"), nullable=False)
    node_id = Column(None, ForeignKey("node.node_id"), nullable=False)
    weight_id = Column(None, ForeignKey("weight.weight_id"), nullable=False)
    age_lower = Column(Float(), nullable=False)
    age_upper = Column(Float(), nullable=False)
    time_lower = Column(Float(), nullable=False)
    time_upper = Column(Float(), nullable=False)


class Data(Base):
    __tablename__ = "data"

    data_id = Column(Integer(), primary_key=True)
    data_name = Column(String(), unique=True, nullable=False)
    """This is in the docs but not in the code."""

    integrand_id = Column(None, ForeignKey("integrand.integrand_id"), nullable=False)
    density_id = Column(None, ForeignKey("density.density_id"), nullable=False)
    node_id = Column(None, ForeignKey("node.node_id"), nullable=False)
    weight_id = Column(None, ForeignKey("weight.weight_id"), nullable=False)
    hold_out = Column(Integer(), nullable=False)
    """Zero or one for hold outs during fit command"""
    meas_value = Column(Float(), nullable=False)
    meas_std = Column(Float(), nullable=False)
    eta = Column(Float(), nullable=True)
    nu = Column(Float(), nullable=True)
    age_lower = Column(Float(), nullable=False)
    age_upper = Column(Float(), nullable=False)
    time_lower = Column(Float(), nullable=False)
    time_upper = Column(Float(), nullable=False)


class Option(Base):
    __tablename__ = "option"

    option_id = Column(Integer(), primary_key=True)
    option_name = Column(String(), unique=True)
    option_value = Column(String(), nullable=False)


class Constraint(Base):
    """Greg shows this table, but it's not in Brad's docs."""

    __tablename__ = "constraint_table"

    constraint_id = Column(Integer(), primary_key=True)
    integrand_id = Column(Integer(), nullable=False)
    density_id = Column(Integer(), nullable=False)
    node_id = Column(Integer(), nullable=False)
    weight_id = Column(Integer(), nullable=False)
    hold_out = Column(Integer(), nullable=False)
    meas_value = Column(Float(), nullable=False)
    meas_std = Column(Float(), nullable=False)
    age_lower = Column(Float(), nullable=False)
    age_upper = Column(Float(), nullable=False)
    time_lower = Column(Float(), nullable=False)
    time_upper = Column(Float(), nullable=False)


class CascadeOption(Base):
    """
    Greg shows this table, but it's not in Brad's docs.
    """

    __tablename__ = "cascade_option_table"

    cascade_option_id = Column(Integer(), primary_key=True)
    cascade_option_name = Column(String(), unique=True, nullable=False)
    cascade_option_value = Column(String(), nullable=False)


class DataSubset(Base):
    """
    Output, identifies which rows of the data table are included in
    the fit data subset table.
    """

    __tablename__ = "data_subset"
    __readonly__ = True

    data_subset_id = Column(Integer(), primary_key=True)
    data_id = Column(None, ForeignKey("data.data_id"), nullable=False)


class DependVar(Base):
    """
    Output, created by the depend command. Diagnostic to see whether the model
    has more variables than necessary.
    """

    __tablename__ = "depend_var_table"
    __readonly__ = True

    depend_var_id = Column(Integer(), primary_key=True)
    data_depend = Column(Integer(), nullable=False)
    prior_depend = Column(Integer(), nullable=False)


class FitVar(Base):
    """
    Output, contains maximum likelihood estimate for the model variables
    corresponding to the data table measured value. A new fit var table
    is created each time the fit command runs.
    """

    __tablename__ = "fit_var_table"
    __readonly__ = True

    fit_var_id = Column(Integer(), primary_key=True)
    fit_var_value = Column(Float(), nullable=False)
    residual_value = Column(Float(), nullable=False)
    residual_dage = Column(Float(), nullable=False)
    residual_dtime = Column(Float(), nullable=False)
    lagrange_value = Column(Float(), nullable=False)
    lagrange_dage = Column(Float(), nullable=False)
    lagrange_dtime = Column(Float(), nullable=False)


class FitDataSubset(Base):
    """
    Output, compares the model and data for the model variables corresponding
    to a fit command. A new fit data subset table is created each time
    the fit command runs.
    """

    __tablename__ = "fit_data_subset_table"
    __readonly__ = True

    fit_data_subset_id = Column(Integer(), primary_key=True)
    avg_integrand = Column(Float(), nullable=False)
    weighted_residual = Column(Float(), nullable=False)


class SampleIndex(Base):
    """
    Output, The sample command creates this table with one optimal estimate
    of the model variable. For each valid simulate index in the simulate
    table, there is an equal sample index in the table with the optimal
    variables corresponding to the measurement.
    """

    __tablename__ = "sample_index_table"

    sample_id = Column(Integer(), primary_key=True)
    sample_index = Column(Integer(), nullable=False)
    var_id = Column(Integer(), nullable=False)
    var_value = Column(Float(), nullable=False)


class Predict(Base):
    """
    Output, model predictions for the average integrand.
    """

    __tablename__ = "predict"
    __readonly__ = True

    predict_id = Column(Integer(), primary_key=True)
    sample_index = Column(Integer(), nullable=False)
    avgint_id = Column(Integer(), nullable=False)
    avg_integrand = Column(Float(), nullable=False)


class ScaleVar(Base):
    """
    Output, the fixed effects are scaled using the eta in the corresponding
    priors. The fixed effects objective and constraints are scaled using their
    values at the model_variables corresponding to the scale_var table.
    The init_command creates a scale_var table using the mean of the priors
    for the model variables values. The set_command can also be used to change
    the start_var table. This table may also be created directly by the user
    (with the aid of the var_table ).
    """

    __tablename__ = "scale_var_table"

    scale_var_id = Column(Integer(), primary_key=True)
    scale_var_value = Column(Float(), nullable=False)


class StartVar(Base):
    """
    Output, the start var table contains one value for each model variable.
    The init_command creates a start_var table using the mean of the priors
    for the model variables values. The set_command can also be used to
    create a start_var table. This table may also be created directly by the
    user (with the aid of the var_table ).
    """

    __tablename__ = "start_var_table"

    start_var_id = Column(Integer(), primary_key=True)
    start_var_value = Column(Float(), nullable=False)


class TruthVar(Base):
    """
    Output, the set command can be used to create a truth variable.
    """

    __tablename__ = "truth_var_table"

    truth_var_id = Column(Integer(), primary_key=True)
    truth_var_value = Column(Float(), nullable=False)


class Simulate(Base):
    """Output"""

    __tablename__ = "simulate_table"

    simulate_id = Column(Integer(), primary_key=True)
    simulate_index = Column(Integer(), nullable=False)
    data_subset_id = Column(None, ForeignKey("data_subset_table.data_subset_id"), nullable=False)
    simulate_value = Column(Float(), nullable=False)  # Greg's has meas_value
    simulate_delta = Column(Float(), nullable=False)  # Greg's has meas_std


class Var(Base):
    """Output"""

    __tablename__ = "var_table"

    var_id = Column(Integer(), primary_key=True)
    var_type = Column(String(), nullable=False)
    smooth_id = Column(Integer(), nullable=False)
    age_id = Column(Integer(), nullable=False)
    time_id = Column(Integer(), nullable=False)
    node_id = Column(Integer(), nullable=False)
    rate_id = Column(Integer(), nullable=False)
    integrand_id = Column(Integer(), nullable=False)
    covariate_id = Column(Integer(), nullable=False)


_TYPE_MAP = {str: String, int: Integer, float: Float}


def add_columns_to_avgint_table(column_identifiers):
    """
    Args:
        column_identifiers: dict(name -> type) where type
            is one of int, float, str
    """
    LOGGER.debug("Adding columns to avgint table {}".format(list(column_identifiers.keys())))
    for name, python_type in column_identifiers.items():
        AvgInt.__table__.append_column(Column(name, _TYPE_MAP[python_type]()))


def add_columns_to_data_table(column_identifiers):
    """
    Args:
        column_identifiers: dict(name -> type) where type
            is one of int, float, str
    """
    LOGGER.debug("Adding columns to data table {}".format(list(column_identifiers.keys())))
    for name, python_type in column_identifiers.items():
        Data.__table__.append_column(Column(name, _TYPE_MAP[python_type]()))
