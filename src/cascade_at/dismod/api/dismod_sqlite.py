"""
Creates a file for Dismod-AT to read.

The file format is sqlite3. This uses a local mapping of database tables
to create it and add tables.

The object wrapper makes Pandas data frames. They get passed here
and validated. Then Pandas uses to_csv to write them to the sqlalchemy
engine, which uses the metadata wrapper (and its custom conversions)
to write them to a very specific format that Dismod-AT is able to read.
"""
from copy import deepcopy
from textwrap import dedent
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype
from sqlalchemy import Enum, Integer, Float
from sqlalchemy import create_engine
from sqlalchemy.exc import StatementError

from cascade_at.core.log import get_loggers
from cascade_at.core.errors import DismodFileError
from cascade_at.dismod.api.table_metadata import Base, add_columns_to_table

LOG = get_loggers(__name__)


def get_engine(file_path):
    if file_path is not None:
        full_path = file_path.expanduser().absolute()
        engine = create_engine("sqlite:///{}".format(str(full_path)))
    else:
        engine = create_engine("sqlite:///:memory:", echo=False)
    return engine


class DismodSQLite:
    """
    Responsible for creation of a Dismod-AT file.

    This class checks the type of all columns. It doesn't check that the
    model is correct::

    Each Dismod table has a primary key named with ``<table>_id``.
    When reading a Pandas dataframe from the file, it will have this
    column as a separate column. When writing a Pandas dataframe to
    the file, if the column isn't present, the index of the dataframe
    will be converted to be the primary key column.

    The arguments for ``avgint_columns`` and ``data_columns`` add columns
    to the avgint and data tables. These arguments are dictionaries from
    column name to column type.

    This uses a deep copy of the metadata module so that, when it adds columns
    to tables, it doesn't affect the module itself.

    Example:
    >>> from pathlib import Path
    >>> path = Path('test.db')
    >>> dm = DismodSQLite(path)
    >>> dm.create_tables()

    >>> data = dm.read_table('data')
    >>> time = pd.DataFrame({'time': [1997, 2005, 2017]})
    >>> dm.write_table('time', time)
    """

    def __init__(self, path: Union[str, Path]):
        """
        Initiates an SQLite reader from the path.

        Arguments
        =========
        path
            A string or Path pointing to the DisMod database file.
        """
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        LOG.debug(f"Creating an engine at {path.absolute()}.")
        self.engine = get_engine(path)
        self._metadata = deepcopy(Base.metadata)
        self._table_definitions = self._metadata.tables
        LOG.debug(f"dmfile tables {self._table_definitions.keys()}")

    def create_tables(self, tables=None):
        """
        Make all of the tables in the metadata.
        """
        LOG.debug(f"Creating table subset {tables}")
        Base.metadata.create_all(self.engine, tables, checkfirst=False)

    def update_table_columns(self, table_name, table):
        """
        Updates the table columns with additional columns like
        "c_" which are comments and "x_" which are covariates.
        """
        table_definition = self._table_definitions[table_name]
        new_columns = table.columns.difference(table_definition.c.keys())
        new_column_types = {c: table.dtypes[c] for c in new_columns}

        allows_covariates = table_definition.name in ["avgint", "data"]

        good_prefixes = ["c_"]
        if allows_covariates:
            good_prefixes.append("x_")
        bad_column_names = [c for c in new_columns if c[:2] not in good_prefixes]
        if bad_column_names:
            msg = f"""
            Table '{table_definition.name}' has these columns {list(table_definition.c.keys())}.
            It allows additional comment columns, which must start 'c_'."""
            if allows_covariates:
                msg += " In addition it allows covariate columns, which must start with 'x_'."
            msg += f" You supplied columns that don't meet those requirements: {bad_column_names}"

            raise ValueError(dedent(msg))

        add_columns_to_table(table_definition, new_column_types)

    def read_table(self, table_name):
        """
        Read a table from the database in engine specified.
        """
        return pd.read_sql_table(table_name=table_name, con=self.engine)

    def write_table(self, table_name, table):
        """
        Writes a table to the database in the engine specified.

        Parameters:
            table_name (str): the name of the table to write to
            table (pd.DataFrame): data frame to write
        """
        table_definition = self._table_definitions[table_name]

        extra_columns = set(table.columns.difference(table_definition.c.keys()))
        if extra_columns:
            self.update_table_columns(table_name, table)

        # Force the table to have the dismod-required columns
        dtypes = {k: v.type for k, v in table_definition.c.items()}
        id_column = f"{table_name}_id"
        if id_column not in table:
            table[id_column] = table.reset_index(drop=True).index
        table = pd.DataFrame(table, columns = dtypes.keys())

        self._validate_data(table_definition, table)

        try:
            table = table.set_index(id_column)
            table.index = table.index.astype(np.int64)
        except ValueError as ve:
            raise ValueError(f"Cannot convert {table_name}.{table_name}_id to index") from ve
        try:
            LOG.debug(f"Writing table {table_name} rows {len(table)} types {dtypes}")
            table.index.name = None
            table.to_sql(
                name=table_name,
                con=self.engine,
                index_label=id_column,
                if_exists="replace",
                dtype=dtypes
            )
        except StatementError:
            raise

    def empty_table(self, table_name, extra_columns=None):
        """
        Initializes an empty table for table_name.
        """
        table_definition = self._table_definitions[table_name]
        df = pd.DataFrame({
            k: pd.Series(dtype=v.type.python_type) for k, v in table_definition.c.items()
        })
        if extra_columns:
            extras = pd.DataFrame({k: pd.Series() for k in extra_columns})
            df = pd.concat([df, extras], axis=1)
        return df

    def _validate_data(self, table_definition, data):
        """Validates that the dtypes in data match the expected types in the
        table_definition.
        Pandas makes this difficult because DataFrames with no length have
        Object type, and those with nulls become float type.

        Dismod-AT has its own set of rules about representation of null values.

         * For a text column, a missing value is an empty string, ``""``.
         * For an integer column, a missing value is the minimum integer,
           but no integer value should ever be missing.
         * For a float column, infinity is the maximum float value,
           which is ``10e318`` or minimum, which is ``-10e318`` according to
           Dismod-AT's arbitrary version of calculating this.
        """
        if len(data) == 0:
            # Length zero columns get converted on write.
            return

        columns_checked = set()

        for column_name, column_definition in table_definition.c.items():
            if column_name in data:
                expected_type = self._expected_type(column_definition)
                is_nullable_numeric = (column_definition.nullable and
                                       expected_type in [int, float])
                if is_nullable_numeric:
                    data[column_name] = data[column_name].fillna(value=np.nan)
                actual_type = data[column_name].dtype
                is_pandas_extension = isinstance(actual_type, ExtensionDtype)
                if expected_type is int:
                    self._check_int_type(actual_type, column_name,
                                         is_pandas_extension, table_definition)
                elif expected_type is float:
                    self._check_float_type(actual_type, column_name,
                                           table_definition)
                elif expected_type is str:
                    self._check_str_type(actual_type, column_name, data,
                                         table_definition)
                else:
                    raise RuntimeError(f"Unexpected type from column "
                                       f"definitions: {expected_type}.")
            elif not (column_definition.primary_key or
                      column_definition.nullable):
                raise DismodFileError(f"Missing column in data for table "
                                      f"'{table_definition.name}': "
                                      f"'{column_name}'")
            columns_checked.add(column_name)

        extra_columns = set(data.columns).difference(table_definition.c.keys())
        if extra_columns:
            raise DismodFileError(f"extra columns in data for table "
                                  f"'{table_definition.name}': {extra_columns}"
                                  )

    @staticmethod
    def _expected_type(column_definition):
        """Column definitions contain type information, and this augments those with rules."""
        try:
            expected_type = column_definition.type.python_type
        except NotImplementedError:
            # Custom column definitions can lack a type.
            # We use custom column definitions for primary keys of type int.
            expected_type = int
        if issubclass(expected_type, Enum):
            # This is an Enum type column, I'm making the simplifying assumption
            # that those will always be string type
            expected_type = str
        return expected_type

    @staticmethod
    def _check_int_type(actual_type, column_name, is_pandas_extension, table_definition):
        if is_pandas_extension:
            if actual_type.is_dtype(pd.Int64Dtype()):
                return
            else:
                raise DismodFileError(
                    f"column '{column_name}' in data for table '{table_definition.name}' must be integer"
                )
        else:
            # Permit np.float because an int column with a None is cast to float.
            # Same for object. This is cast on write.
            # Because we use metadata, this will be converted for us to int when it is written.
            allowed = [np.integer, np.floating]
            if not any(np.issubdtype(actual_type, given_type) for given_type in allowed):
                raise DismodFileError(
                    f"column '{column_name}' in data for table '{table_definition.name}' must be integer"
                )

    @staticmethod
    def _check_float_type(actual_type, column_name, table_definition):
        if not np.issubdtype(actual_type, np.number):
            raise DismodFileError(
                f"column '{column_name}' in data for table '{table_definition.name}' must be numeric"
            )

    @staticmethod
    def _check_str_type(actual_type, column_name, data, table_definition):
        if len(data) > 0:
            correct = data[column_name].dtype == np.dtype('O')
            if not correct:
                raise DismodFileError(
                    f"column '{column_name}' in data for table '{table_definition.name}' must be string "
                    f"but type is {actual_type}."
                )
        else:
            pass  # Will convert to string on write of empty rows.
