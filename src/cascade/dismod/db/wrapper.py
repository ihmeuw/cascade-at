"""
Creates a file for Dismod-AT to read.

The file format is sqlite3. This uses a local mapping of database tables
to create it and add tables.

The object wrapper makes Pandas Dataframes. They get passed here
and validated. Then Pandas uses to_csv to write them to the sqlalchemy
engine, which uses the metadata wrapper (and its custom conversions)
to write them to a very specific format that Dismod-AT is able to read.
"""
from copy import deepcopy
from textwrap import dedent

import numpy as np
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype
from networkx import DiGraph
from networkx.algorithms.dag import lexicographical_topological_sort
from sqlalchemy import Integer, String, Float, Enum
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, StatementError
from sqlalchemy.sql import select, text

from cascade.core import getLoggers
from cascade.dismod.db import DismodFileError
from cascade.dismod.db.metadata import Base, add_columns_to_table

CODELOG, MATHLOG = getLoggers(__name__)


def get_engine(file_path):
    if file_path is not None:
        full_path = file_path.expanduser().absolute()
        engine = create_engine("sqlite:///{}".format(str(full_path)))
    else:
        engine = create_engine("sqlite:///:memory:", echo=False)
    return engine


def _validate_data(table_definition, data):
    """Validates that the dtypes in data match the expected types in the table_definition.
    Pandas makes this difficult because DataFrames with no length have Object type,
    and those with nulls become float type.

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
            actual_type = data[column_name].dtype
            is_pandas_extension = isinstance(actual_type, ExtensionDtype)
            expected_type = _expected_type(column_definition)

            if expected_type is int:
                _check_int_type(actual_type, column_name, is_pandas_extension, table_definition)
            elif expected_type is float:
                _check_float_type(actual_type, column_name, table_definition)
            elif expected_type is str:
                _check_str_type(actual_type, column_name, data, table_definition)
            else:
                raise RuntimeError(f"Unexpected type from column definitions: {expected_type}.")
        elif not (column_definition.primary_key or column_definition.nullable):
            raise DismodFileError(f"Missing column in data for table '{table_definition.name}': '{column_name}'")
        columns_checked.add(column_name)

    extra_columns = set(data.columns).difference(table_definition.c.keys())
    if extra_columns:
        raise DismodFileError(f"extra columns in data for table '{table_definition.name}': {extra_columns}")


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


def _check_float_type(actual_type, column_name, table_definition):
    if not np.issubdtype(actual_type, np.number):
        raise DismodFileError(
            f"column '{column_name}' in data for table '{table_definition.name}' must be numeric"
        )


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


def _ordered_by_foreign_key_dependency(schema, tables_to_write):
    """
    Iterator that walks through tables in the correct order for foreign
    keys to be written before those that depend upon them.

    Args:
        schema (sqlalchemy.metadata): Metadata about the tables.
        tables_to_write (iterable[str]): An iterable of table names as string.

    Returns:
        Iterates through the tables in an order that is safe for writing.
    """
    # TODO: This should subset schema.sorted_tables to those tables in tables_to_write.
    dependency_graph = DiGraph()

    unwanted_tables = set(tables_to_write) - set(schema.tables.keys())
    if unwanted_tables:
        raise DismodFileError(f"Asking to write tables not in schema: {unwanted_tables}")

    for scan_name, scan_table in schema.tables.items():
        dependency_graph.add_node(scan_name)
        for foreign_key in scan_table.foreign_keys:
            target_name = foreign_key.target_fullname.split(".")[0]
            dependency_graph.add_edge(target_name, scan_name)

    # The pure topological sort might be faster.
    # Use the full lexicographical sort because it makes testing deterministic.
    for next_table in lexicographical_topological_sort(dependency_graph):
        if next_table in tables_to_write:
            yield next_table


class DismodFile:
    """
    Responsible for creation of a Dismod-AT file.

    This class checks the type of all columns. It doesn't check that the
    model is correct::

        engine = _get_engine(None)
        dm = DismodFile(engine, {"col": float}, {})
        dm_file.time = pd.DataFrame({"time": [1997, 2005, 2017]}))
        time_df = dm_file.time

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
    """

    def __init__(self, engine=None):
        """
        The columns arguments add columns to the avgint and data
        tables.

        Args:
            engine: A sqlalchemy engine
        """
        self.engine = engine
        self._metadata = deepcopy(Base.metadata)
        self._table_definitions = self._metadata.tables
        self._table_data = {}
        self._table_hash = {}
        CODELOG.debug(f"dmfile tables {self._table_definitions.keys()}")

    def create_tables(self, tables=None):
        """
        Make all of the tables in the metadata.
        """
        CODELOG.debug(f"Creating table subset {tables}")
        Base.metadata.create_all(self.engine, tables, checkfirst=False)

    def __dir__(self):
        attributes = list(self._table_definitions.keys())
        attributes.extend(super().__dir__())
        return attributes

    def update_table_columns(self, table_name, table):
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

    def refresh(self, evict_tables=None):
        """ Throw away any un-flushed changes and reread data from disk.
        """
        evicted = list()
        if evict_tables is None:
            evicted = list(self._table_data.keys())
            self._table_data = {}
            self._table_hash = {}
        else:
            to_evict = [evict_tables] if isinstance(evict_tables, str) else evict_tables
            for evict in to_evict:
                if evict in self._table_data:
                    del self._table_data[evict]
                if evict in self._table_hash:
                    del self._table_hash[evict]
                    evicted.append(evict)
        CODELOG.debug(f"Evicted from wrapper: {evicted}")

    def __getattr__(self, table_name):
        if table_name in self._table_data:
            return self._table_data[table_name]
        elif table_name in self._table_definitions:
            read_from_database = False
            table = self._table_definitions[table_name]
            if self.engine is None:
                data = self.empty_table(table.name)
            else:
                try:
                    with self.engine.connect() as conn:
                        data = pd.read_sql_table(table.name, conn)
                        read_from_database = True
                except ValueError as e:
                    if str(e) != f"Table {table.name} not found":
                        raise
                    data = self.empty_table(table.name)

            extra_columns = set(data.columns.difference(table.c.keys()))
            if extra_columns:
                self.update_table_columns(table_name, data)

            data = data.set_index(f"{table_name}_id", drop=False)
            # The hash table defines whether the table is new, and whether
            # a table created is new.
            if read_from_database:
                self._table_hash[table_name] = pd.util.hash_pandas_object(data)
            self._table_data[table_name] = data
            return data
        else:
            raise AttributeError(
                f"Tried to write to {table_name} in Dismod db file but no such table is listed in the schema.")

    def __setattr__(self, table_name, df):
        if table_name in self.__dict__.get("_table_definitions", {}):
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Tried to set table using type {type(df)} instead of a DataFrame")
            if f"{table_name}_id" not in df:
                df = df.assign(**{f"{table_name}_id": df.index})
            self._table_data[table_name] = df
        elif isinstance(df, pd.DataFrame):
            raise KeyError(f"Tried to set table {table_name} but it isn't in the db specification")
        else:
            super().__setattr__(table_name, df)

    def _is_dirty(self, table_name):
        """Tests to see if the table's data has changed in memory since it was last loaded from the database.
        """

        if table_name not in self._table_data:
            return False

        table = self._table_data[table_name]
        try:
            table_hash = pd.util.hash_pandas_object(table)
        except TypeError as te:
            if "mutable" in str(te):
                CODELOG.warning(f"table {table_name} dtypes {table.dtypes}")
            raise DismodFileError(f"The table {table_name} has unexpected value while saving.") from te

        is_new = table_name not in self._table_hash
        if not is_new:
            is_changed = not self._table_hash[table_name].equals(table_hash)
        else:
            is_changed = False

        return is_new or is_changed

    def flush(self):
        """Writes any data in memory to the underlying database. Data which has not been changed since
        it was last written is not re-written.
        """
        if self.engine is None:
            raise RuntimeError("Cannot flush db file tables before an engine is set "
                               "and the engine is None.")
        with self.engine.begin() as connection:
            CODELOG.debug(f"DismodFile has table data for {', '.join(sorted(self._table_data.keys()))}")
            for table_name in _ordered_by_foreign_key_dependency(Base.metadata, self._table_data.keys()):
                if self._is_dirty(table_name):
                    table = self._table_data[table_name]
                    if hasattr(table, "__readonly__") and table.__readonly__:
                        raise DismodFileError(f"Table '{table_name}' is not writable")

                    table_definition = self._table_definitions[table_name]

                    extra_columns = set(table.columns.difference(table_definition.c.keys()))
                    if extra_columns:
                        self.update_table_columns(table_name, table)

                    _validate_data(table_definition, table)

                    # TODO: I'm re-calculating this hash for the sake of having a nice _is_dirty function.
                    # That may be too expensive.
                    table_hash = pd.util.hash_pandas_object(table)

                    # In order to write the <table>_id column as a primary key,
                    # we need to make the index not have a name and be of type
                    # int64. That makes it write as a BIGINT, which the
                    # sqlalchemy compiler turns into a primary key statement.
                    if f"{table_name}_id" in table:
                        table = table.set_index(f"{table_name}_id")
                        try:
                            table.index = table.index.astype(np.int64)
                        except ValueError as ve:
                            raise ValueError(f"Cannot convert {table_name}.{table_name}_id to index") from ve
                    try:
                        dtypes = {k: v.type for k, v in table_definition.c.items()}
                        CODELOG.debug(f"Writing table {table_name} rows {len(table)} types {dtypes}")
                        table.index.name = None
                        table.to_sql(
                            table_name, connection, index_label=f"{table_name}_id", if_exists="replace", dtype=dtypes
                        )
                    except StatementError:
                        raise

                    self._table_hash[table_name] = table_hash
                else:
                    CODELOG.debug(f"{table_name} did not need to be written")

        self._check_column_types_actually_written()

    def _check_column_types_actually_written(self):
        """
        They can be written differently than what you declare in metadata
        because primary keys and joint keys can invoke hidden transformations.
        """
        expect = {"integer": Integer(), "text": String(), "real": Float()}

        with self.engine.begin() as connection:
            for table_name, table_definition in self._table_definitions.items():
                introspect = text(f"PRAGMA table_info([{table_name}]);")
                results = connection.execute(introspect)
                if not results.returns_rows:
                    continue
                table_info = results.fetchall()
                if table_info:
                    in_db = {row[1]: (row[2].lower(), int(row[5])) for row in table_info}
                    for column_name, column_object in table_definition.c.items():
                        if column_name not in in_db:
                            raise RuntimeError(
                                f"Expected a column to be written but it was not there. "
                                f"missing: {table_name}.{column_name}. "
                                f"Columns present {table_info}."
                            )
                        column_type, primary_key = in_db[column_name]
                        if column_name == f"{table_name}_id" and primary_key != 1:
                            raise RuntimeError(
                                f"table {table_name} column {column_name} is not a primary key {primary_key}"
                            )
                        if column_type not in expect:
                            raise RuntimeError(
                                f"A sqlite column type is unexpected: " f"{table_name}.{column_name} {column_type}"
                            )
                        if type(column_object.type) != type(expect[column_type]):  # noqa: E721
                            raise RuntimeError(f"{table_name}.{column_name} got wrong type {column_type}")

                    CODELOG.debug(f"Table {table_name} {table_info}")

    def diagnostic_print(self):
        """
        Print all values to the screen. This isn't as heavily-formatted
        as db2csv.
        """
        with self.engine.connect() as connection:
            for name, table in self._table_definitions.items():
                print(name)
                try:
                    for row in connection.execute(select([table])):
                        print(row)
                except OperationalError:
                    pass  # That table doesn't exist.

    def empty_table(self, table_name):
        """
        Creates a data table that is empty but has the correct types
        for all columns. We make this because, if you create an empty
        data table that has just the correct column names, then the
        primary key will be written as a string.

        Args:
            table_name (str): Must be one of the tables defined by metadata.

        Returns:
            An empty dataframe, but columns have correct types.
        """
        table_definition = self._table_definitions[table_name]
        return pd.DataFrame({k: pd.Series(dtype=v.type.python_type) for k, v in table_definition.c.items()})
