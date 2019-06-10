"""
This describes the tables in the sqlite file that Dismod reads.

Use this interface instead of raw SQL queries becasue

 * It controls the data type for each column. The Sqlite
   db doesn't say what the data type should be in Python.

 * It verifies that the db has what we think it has and
   warns us when database tables change names or when
   columns change names.

 * It lets us change a column name in the db table without
   changing the column name we use to read it. This
   protects us against column name changes, which are
   freuqent.

 * It records which tables depend on which other tables
   which is necessary in order to write Pandas versions
   to the SQL file in the right order.

"""

import numpy as np
import pandas as pd

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Enum, ForeignKey
from sqlalchemy import BigInteger
from sqlalchemy.ext.compiler import compiles

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


# Sqlite matches names to types. Brad checks exact names against a set
# that he chose arbitrarily. This is a way to match his arbitrary choices.
# Sqlite3 type system: https://www.sqlite.org/datatype3.html
# Brad's allowed list: integer, real, text.
# The method is here:
#   http://docs.sqlalchemy.org/en/latest/core/compiler.html?highlight=compiler#module-sqlalchemy.ext.compiler


@compiles(Integer, "sqlite")
def _integer_callback(element, compiler, **kw):
    """Changes INTEGER to integer."""
    return "integer"


@compiles(BigInteger, "sqlite")
def _big_integer_callback(element, compiler, **kw):
    """Changes INTEGER to integer."""
    # The keywords include one called type_expression, which is the column
    # specification. This should tell us whether the type is a primary key,
    # but that's False, even when the type is a primary key. However,
    # sqlalchemy automatically promotes every primary key to be a big integer,
    # so we mark them here as primary keys.
    return "integer primary key"


@compiles(Float, "sqlite")
def _float_callback(element, compiler, **kw):
    """Changes all FLOAT types to real"""
    return "real"


@compiles(String, "sqlite")
def _string_callback(element, compiler, **kw):
    """Changes VARCHAR to text."""
    return "text"


@compiles(Enum, "sqlite")
def _enum_callback(element, compiler, **kw):
    """Changes VARCHAR to text."""
    return "text"


Base = declarative_base()


class Log(Base):
    __tablename__ = "log"

    log_id = Column(Integer(), primary_key=True, autoincrement=True)
    message_type = Column(String(), nullable=True)
    table_name = Column(String(), nullable=True)
    row_id = Column(Integer(), nullable=True)
    unix_time = Column(Integer(), nullable=False)
    message = Column(String(), nullable=True)


class Age(Base):
    __tablename__ = "age"

    age_id = Column(Integer(), primary_key=True, autoincrement=False)
    age = Column(Float, unique=True, nullable=False)


class AgeAvg(Base):
    __tablename__ = "age_avg"

    age_avg_id = Column(Integer(), primary_key=True, autoincrement=False)
    age = Column(Float, unique=True, nullable=False)


class Time(Base):
    __tablename__ = "time"

    time_id = Column(Integer(), primary_key=True, autoincrement=False)
    time = Column(Float, unique=True, nullable=False)


class Integrand(Base):
    """These are the names of the integrands, taken from IntegrandEnum"""

    __tablename__ = "integrand"

    integrand_id = Column(Integer(), primary_key=True, autoincrement=False)
    integrand_name = Column(String(), unique=True, nullable=False)
    """
    Each integrand may appear only once. Unused integrands need not be added.
    """
    minimum_meas_cv = Column(Float())


class Density(Base):
    """Defines names of distributions to use as priors."""

    __tablename__ = "density"

    density_id = Column(Integer(), primary_key=True, autoincrement=False)
    density_name = Column(String(), unique=True, nullable=False)


class Covariate(Base):
    """The names of covariates and some reference values"""

    __tablename__ = "covariate"

    covariate_id = Column(Integer(), primary_key=True, autoincrement=False)
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

    node_id = Column(Integer(), primary_key=True, autoincrement=False)
    node_name = Column(String(), nullable=False, unique=True)
    parent = Column(Integer(), nullable=True)  # Parent is an id in _this_ table.


class Prior(Base):
    __tablename__ = "prior"

    prior_id = Column(Integer(), primary_key=True, autoincrement=False)
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

    weight_id = Column(Integer(), primary_key=True, autoincrement=False)
    weight_name = Column(String(), unique=True)
    n_age = Column(Integer())
    """The number of age values in the smoothing grid. Greater than zero"""
    n_time = Column(Integer())
    """The number of time values in the smoothing grid. Greater than zero"""


class WeightGrid(Base):
    __tablename__ = "weight_grid"

    weight_grid_id = Column(Integer(), primary_key=True, autoincrement=False)
    weight_id = Column(None, ForeignKey("weight.weight_id"), nullable=False)
    age_id = Column(None, ForeignKey("age.age_id"), nullable=False)
    time_id = Column(None, ForeignKey("time.time_id"), nullable=False)
    weight = Column(Float(), nullable=False)
    """This is the weight for this age, time, and weight id."""


class Smooth(Base):
    __tablename__ = "smooth"

    smooth_id = Column(Integer(), primary_key=True, nullable=False, autoincrement=False)
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

    smooth_grid_id = Column(Integer(), primary_key=True, autoincrement=False)
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

    nslist_id = Column(Integer(), primary_key=True, autoincrement=False)
    nslist_name = Column(String(), unique=True, nullable=False)


class NSListPair(Base):
    """Associates a node with a smoothing"""

    __tablename__ = "nslist_pair"

    nslist_pair_id = Column(Integer(), primary_key=True, autoincrement=False)
    nslist_id = Column(None, ForeignKey("nslist.nslist_id"), nullable=False)
    node_id = Column(None, ForeignKey("node.node_id"), nullable=False)
    smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=False)


class Rate(Base):
    __tablename__ = "rate"

    rate_id = Column(Integer(), primary_key=True, autoincrement=False)
    rate_name = Column(String(), nullable=False)
    parent_smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=True)
    """If null, then parent rate is always zero and no model variables are
    allocated for it"""

    child_smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=True)
    child_nslist_id = Column(None, ForeignKey("nslist.nslist_id"), nullable=True)
    """If not null, identifies a list of node and smoothing pairs. The
    node_id for each of the children must appear in the list. The corresponding
    smoothing is used for that child and the rate corresponding to this
    row of the rate table."""


class MulCov(Base):
    """Covariate multipliers and their priors.

    If this is empty, there are no covariate multipliers in the model.
    """

    __tablename__ = "mulcov"

    mulcov_id = Column(Integer(), primary_key=True, autoincrement=False)
    mulcov_type = Column(String(), nullable=False)
    rate_id = Column(None, ForeignKey("rate.rate_id"), nullable=True)
    """Determines the rate that this covariate and multiplier affects.
    If mulcov_type is of type meas_value or meas_std, this must be null."""
    integrand_id = Column(None, ForeignKey("integrand.integrand_id"), nullable=True)
    covariate_id = Column(None, ForeignKey("covariate.covariate_id"), nullable=False)
    smooth_id = Column(None, ForeignKey("smooth.smooth_id"), nullable=True)
    """If this is null, the covariate multiplier is always zero and no
    model_variables are allocated for it."""


class AvgInt(Base):
    """
    Each entry in avgint asks Dismod to calculate a value for this measure,
    age, and time.
    """

    __tablename__ = "avgint"

    avgint_id = Column(Integer(), primary_key=True, autoincrement=False)
    integrand_id = Column(None, ForeignKey("integrand.integrand_id"), nullable=False)
    node_id = Column(None, ForeignKey("node.node_id"), nullable=False)
    weight_id = Column(None, ForeignKey("weight.weight_id"), nullable=False)
    age_lower = Column(Float(), nullable=False)
    age_upper = Column(Float(), nullable=False)
    time_lower = Column(Float(), nullable=False)
    time_upper = Column(Float(), nullable=False)


class Data(Base):
    """
    These are input observations of demographic rates, prevalence, initial
    prevalence. They can also be constraints, entered with hold_out=1
    and as point values in age and time, instead of having different upper
    and lower limits.
    """

    __tablename__ = "data"

    data_id = Column(Integer(), primary_key=True, autoincrement=False)
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

    option_id = Column(Integer(), primary_key=True, autoincrement=False)
    option_name = Column(String(), unique=True)
    option_value = Column(String(), nullable=False)


class ExecutionData(Base):
    __tablename__ = "c_execution_data"

    c_execution_data_id = Column(Integer(), primary_key=True, autoincrement=False)
    key = Column(String(), unique=True)
    value = Column(String(), nullable=False)


class DataSubset(Base):
    """
    Output, identifies which rows of the data table are included in
    the fit data subset table.
    """

    __tablename__ = "data_subset"
    __readonly__ = True

    data_subset_id = Column(Integer(), primary_key=True, autoincrement=False)
    data_id = Column(Integer(), ForeignKey("data.data_id"), nullable=False)


class DependVar(Base):
    """
    Output, created by the depend command. Diagnostic to see whether the model
    has more variables than necessary.
    """

    __tablename__ = "depend_var"
    __readonly__ = True

    depend_var_id = Column(Integer(), primary_key=True, autoincrement=False)
    data_depend = Column(Integer(), nullable=False)
    prior_depend = Column(Integer(), nullable=False)


class FitVar(Base):
    """
    Output, contains maximum likelihood estimate for the model variables
    corresponding to the data table measured value. A new fit var table
    is created each time the fit command runs.
    """

    __tablename__ = "fit_var"
    __readonly__ = True

    fit_var_id = Column(Integer(), primary_key=True, autoincrement=False)
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

    __tablename__ = "fit_data_subset"
    __readonly__ = True

    fit_data_subset_id = Column(Integer(), primary_key=True, autoincrement=False)
    avg_integrand = Column(Float(), nullable=False)
    weighted_residual = Column(Float(), nullable=False)


class Sample(Base):
    """
    Output, The sample command creates this table with one optimal estimate
    of the model variable. For each valid simulate index in the simulate
    table, there is an equal sample index in the table with the optimal
    variables corresponding to the measurement.
    """

    __tablename__ = "sample"

    sample_id = Column(Integer(), primary_key=True, autoincrement=False)
    sample_index = Column(Integer(), nullable=False)
    var_id = Column(Integer(), nullable=False)
    var_value = Column(Float(), nullable=False)


class Predict(Base):
    """
    Output, model predictions for the average integrand.
    """

    __tablename__ = "predict"
    __readonly__ = True

    predict_id = Column(Integer(), primary_key=True, autoincrement=False)
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

    __tablename__ = "scale_var"

    scale_var_id = Column(Integer(), primary_key=True, autoincrement=False)
    scale_var_value = Column(Float(), nullable=False)


class StartVar(Base):
    """
    Output, the start var table contains one value for each model variable.
    The init_command creates a start_var table using the mean of the priors
    for the model variables values. The set_command can also be used to
    create a start_var table. This table may also be created directly by the
    user (with the aid of the var_table ).
    """

    __tablename__ = "start_var"

    start_var_id = Column(Integer(), primary_key=True, autoincrement=False)
    start_var_value = Column(Float(), nullable=False)


class TruthVar(Base):
    """
    Output, the set command can be used to create a truth variable.
    """

    __tablename__ = "truth_var"

    truth_var_id = Column(Integer(), primary_key=True, autoincrement=False)
    truth_var_value = Column(Float(), nullable=False)


class Simulate(Base):
    """Output"""

    __tablename__ = "simulate"

    simulate_id = Column(Integer(), primary_key=True, autoincrement=False)
    simulate_index = Column(Integer(), nullable=False)
    data_subset_id = Column(None, ForeignKey("data_subset_table.data_subset_id"), nullable=False)
    simulate_value = Column(Float(), nullable=False)  # Greg's has meas_value
    simulate_delta = Column(Float(), nullable=False)  # Greg's has meas_std


class DataSim(Base):
    """The simulate command is a kind of bootstrapping, and this table
    has replacements for the meas_value in the data table in the column
    called ``data_sim_value``."""
    __tablename__ = "data_sim"

    data_sim_id = Column(Integer(), primary_key=True, autoincrement=False)
    simulate_index = Column(Integer(), nullable=False)
    data_subset_id = Column(Integer(), ForeignKey("data_subset_table.data_subset_id"), nullable=False)
    data_sim_value = Column(Float(), nullable=False)
    data_sim_delta = Column(Float(), nullable=False)
    data_sim_stdcv = Column(Float(), nullable=False)


class PriorSim(Base):
    __tablename__ = "prior_sim"

    prior_sim_id = Column(Integer(), primary_key=True, autoincrement=False)
    simulate_index = Column(Integer(), nullable=False)
    var_id = Column(Integer(), nullable=False)
    prior_sim_value = Column(Float(), nullable=True)
    prior_sim_dage = Column(Float(), nullable=True)
    prior_sim_dtime = Column(Float(), nullable=True)


class Var(Base):
    """Output"""

    __tablename__ = "var"

    var_id = Column(Integer(), primary_key=True, autoincrement=False)
    var_type = Column(String(), nullable=False)
    smooth_id = Column(Integer(), nullable=False)
    age_id = Column(Integer(), nullable=False)
    time_id = Column(Integer(), nullable=False)
    node_id = Column(Integer(), nullable=True)
    rate_id = Column(Integer(), nullable=False)
    integrand_id = Column(Integer(), nullable=True)
    covariate_id = Column(Integer(), nullable=True)
    mulcov_id = Column(Integer(), nullable=True)


_TYPE_MAP = {
    np.dtype("O"): String,
    str: String,
    int: Integer,
    float: Float,
    np.dtype("int64"): Integer,
    pd.Int64Dtype(): Integer,  # nullable integer type.
    np.dtype("float64"): Float,
}


def add_columns_to_table(table, column_identifiers):
    """
    Args:
        column_identifiers: dict(name -> type) where type
            is one of int, float, str
    """
    CODELOG.debug(f"Adding columns to {table.name} table {list(column_identifiers.keys())}")
    for name, python_type in column_identifiers.items():
        table.append_column(Column(name, _TYPE_MAP[python_type]()))
