"""
Creates a file for Dismod-AT to read.

The file format is sqlite3. This uses a local mapping of database tables
to create it and add tables.
"""
import logging
from enum import Enum

from networkx import DiGraph
from networkx.algorithms.dag import lexicographical_topological_sort
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.sql import select
from sqlalchemy.exc import OperationalError, StatementError

from .metadata import Base, add_columns_to_avgint_table, add_columns_to_data_table, DensityEnum
from . import DismodFileError


LOGGER = logging.getLogger(__name__)


def _get_engine(file_path):
    if file_path is not None:
        full_path = file_path.expanduser().absolute()
        engine = create_engine("sqlite:///{}".format(str(full_path)))
    else:
        engine = create_engine("sqlite:///:memory:", echo=False)
    return engine


def _validate_data(table_definition, data):
    """Validates that the dtypes in data match the expected types in the table_definition
    """
    columns_checked = set()

    for column_name, column_definition in table_definition.c.items():
        if column_name in data:
            actual_type = data[column_name].dtype
            expected_type = column_definition.type.python_type

            if issubclass(expected_type, Enum):
                # This is an Enum type column, I'm making the simplifying assumption
                # that those will always be string type
                expected_type = str

            if expected_type is int:
                if not np.issubdtype(actual_type, np.integer):
                    raise DismodFileError(
                        f"column '{column_name}' in data for table '{table_definition.name}' must be integer"
                    )
            elif expected_type is float:
                if not np.issubdtype(actual_type, np.number):
                    raise DismodFileError(
                        f"column '{column_name}' in data for table '{table_definition.name}' must be numeric"
                    )
            elif expected_type is str:
                correct = actual_type != np.dtype("O")
                if len(data) > 0:
                    actual_type = type(data[column_name][0])
                    correct = np.issubdtype(actual_type, np.str_)

                if not correct:
                    raise DismodFileError(
                        f"column '{column_name}' in data for table '{table_definition.name}' must be string"
                    )
        elif not (column_definition.primary_key or column_definition.nullable):
            raise DismodFileError(f"Missing column in data for table '{table_definition.name}': '{column_name}'")
        columns_checked.add(column_name)

    extra_columns = set(data.columns).difference(table_definition.c.keys())
    if extra_columns:
        raise DismodFileError(f"extra columns in data for table '{table_definition.name}': {extra_columns}")


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
    dependency_graph = DiGraph()

    if set(tables_to_write) - set(schema.tables.keys()):
        raise ValueError("Asking to write tables not in schema")

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
        dm_file.add("time", pd.DataFrame({"time": [1997, 2005, 2017]}))
        time_df = dm_file.time

    The arguments for ``avgint_columns`` and ``data_columns`` add columns
    to the avgint and data tables. These arguments are dictionaries from
    column name to column type.
    """

    def __init__(self, engine, avgint_columns, data_columns):
        """
        The columns arguments add columns to the avgint and data
        tables.

        Args:
            engine: A sqlalchemy engine
            avgint_columns (dict): From columns to types.
            data_columns (dict): From columns to types.
        """
        self.engine = engine
        self._table_definitions = Base.metadata.tables
        self._table_data = {}
        self._table_hash = {}
        add_columns_to_avgint_table(avgint_columns)
        add_columns_to_data_table(data_columns)
        LOGGER.debug(f"dmfile tables {self._table_definitions.keys()}")

    def create_tables(self):
        """
        Make all of the tables in the metadata.
        """
        # TODO: we only actually need to make a subset of the tables
        Base.metadata.create_all(self.engine)

    def make_densities(self):
        """
        Dismod documentation says all densities should be in the file,
        so this puts them all in.
        """
        self.density = pd.DataFrame({"density_name": [x.name for x in DensityEnum]})

    def __getattr__(self, table_name):
        if table_name in self._table_data:
            return self._table_data[table_name]
        elif table_name in self._table_definitions:
            table = self._table_definitions[table_name]
            with self.engine.connect() as conn:
                data = pd.read_sql_query(select([table]), conn)
            data = data.set_index(f"{table_name}_id")
            self._table_hash[table_name] = pd.util.hash_pandas_object(data)
            self._table_data[table_name] = data
            return data
        else:
            raise AttributeError(f"No such table {table_name}")

    def __setattr__(self, table_name, df):
        if table_name in self.__dict__.get("_table_definitions", {}):
            self._table_data[table_name] = df
        else:
            super().__setattr__(table_name, df)

    def _is_dirty(self, table_name):
        """Tests to see if the table's data has changed in memory since it was last loaded from the database.
        """

        if table_name not in self._table_data:
            return False

        table = self._table_data[table_name]
        table_hash = pd.util.hash_pandas_object(table)

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
        with self.engine.begin() as connection:
            for table_name in _ordered_by_foreign_key_dependency(Base.metadata, self._table_data.keys()):
                if self._is_dirty(table_name):
                    table = self._table_data[table_name]
                    if hasattr(table, "__readonly__") and table.__readonly__:
                        raise DismodFileError(f"Table '{table_name}' is not writable")

                    table_definition = self._table_definitions[table_name]
                    _validate_data(table_definition, table)
                    try:
                        table.to_sql(
                            table_name,
                            connection,
                            index_label=table_name + "_id",
                            if_exists="replace",
                            dtype={k: v.type for k, v in table_definition.c.items()},
                        )
                    except StatementError as e:
                        raise

                    # TODO: I'm re-calculating this hash for the sake of having a nice _is_dirty function.
                    # That may be too expensive.
                    table_hash = pd.util.hash_pandas_object(table)
                    self._table_hash[table_name] = table_hash

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
