import shelve
import sqlite3
from pathlib import Path

import pandas as pd

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class FileEntity:
    """Responsible for making a path that is writable for a file.

    Args:
        relative_path (Path|str): Path to the file, relative to whatever
            directory the execution context says this ``location_id``
            should use.
        location_id (int): The location for which this file is written.
        sex (str): One of male, female, both.
    """
    def __init__(self, relative_path, location_id=None, sex=None):
        # If location_id isn't specified, it's the same location as the reader.
        self.relative_path = Path(relative_path)
        self.location_id = location_id
        self.sex = sex

    def path(self, execution_context):
        """Return a full file path to the file, given the current context."""
        base_directory = execution_context.model_base_directory(
            self.location_id, self.sex)
        full_path = base_directory / self.relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def remove(self, execution_context):
        """Delete, unlink, remove the file. No error if it doesn't exist."""
        path = self.path(execution_context)
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # OK if it didn't exist


class DbFile(FileEntity):
    """Responsible for validating a Dismod-AT db file.
    This validates that all tables with the given names exist.

    Args:
        relative_path (Path|str): Path to the file, relative to whatever
            directory the execution context says this ``location_id``
            should use.
        location_id (int): The location for which this file is written.
        sex (str): One of male, female, both.
        required_tables (Set[str]): A set of table names.
    """
    def __init__(self, relative_path, location_id=None, sex=None, required_tables=None):
        super().__init__(relative_path, location_id, sex)
        self._tables = set(required_tables) if required_tables else None

    def validate(self, execution_context):
        """Validate by checking which tables exist because this establishes
        which steps have been run. Doesn't look inside the tables.

        Returns:
            None, on success, or a dictionary of expected and found, on error.
        """
        if self._tables is None:
            return
        path = self.path(execution_context)
        if not path.exists():
            return dict(found=set(), expected=self._tables)
        with sqlite3.connect(path) as conn:
            result = conn.execute("select name from sqlite_master where type='table'")
            tables = {x[0] for x in result}

        if self._tables - tables:
            return dict(found=tables, expected=self._tables)

    def mock(self, execution_context):
        """Creates a sqlite3 file with the given tables. They don't correspond
        to the columns that are actually expected."""
        if self._tables is None:
            self.path(execution_context).open("w").close()
        path = self.path(execution_context)
        with sqlite3.connect(path) as conn:
            for table in self._tables:
                conn.execute(f"CREATE TABLE {table} (key text, value text)")
            conn.commit()

    def remove(self, execution_context):
        path = self.path(execution_context)
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # OK if it didn't exist


class PandasFile(FileEntity):
    """Responsible for validating a Pandas file.

    Args:
        relative_path (Path|str): Path to the file, relative to whatever
            directory the execution context says this ``location_id``
            should use.
        location_id (int): The location for which this file is written.
        sex (str): One of male, female, both.
        required_frames (Dict[str,set]): Map from the name of the dataset,
            as specified by the Pandas ``key`` argument, to a list of columns
            that should be in that dataset.
    """
    def __init__(self, relative_path, location_id=None, sex=None, required_frames=None):
        super().__init__(relative_path, location_id, sex)
        required_frames = required_frames if required_frames else dict()
        self._columns = {key: set(cols) for (key, cols) in required_frames.items()}

    def validate(self, execution_context):
        """
        Returns:
            None, on success, or a dictionary of expected and found, on error.
        """
        path = self.path(execution_context)
        if not path.exists():
            return dict(found=set(), expected=self._columns)
        errors = dict()
        for key, cols in self._columns.items():
            try:
                df = pd.read_hdf(path, key=key)
                if cols != set(df.columns):
                    errors[key] = dict(found=df.columns, expected=cols)
            except KeyError as key:
                errors[key] = dict(found=set(), expected=cols)
        return errors if errors else None

    def mock(self, execution_context):
        path = self.path(execution_context)
        for key, cols in self._columns.items():
            df = pd.DataFrame(columns=cols)
            df.to_hdf(path, key=key)


class ShelfFile(FileEntity):
    """Responsible for validating a Python shelf file.

    Args:
        relative_path (Path|str): Path to the file, relative to whatever
            directory the execution context says this ``location_id``
            should use.
        location_id (int): The location for which this file is written.
        sex (str): One of male, female, both.
        required_keys (Set[str]): String names of variables to find in the file.
    """
    def __init__(self, relative_path, location_id=None, sex=None, required_keys=None):
        super().__init__(relative_path, location_id, sex)
        self._keys = set(required_keys) if required_keys else set()

    def validate(self, execution_context):
        """
        Validates that there are variables named after the required keys.
        Returns:
            None, on success, or a dictionary of expected and found, on error.
        """
        path = self.path(execution_context)
        search_name = path.parent / (path.name + ".dat")
        if not search_name.exists():
            CODELOG.debug(f"Shelf path doesn't exist {path}")
            return dict(expected=self._keys, found=set())
        if self._keys:
            with shelve.open(str(path)) as db:
                in_file = set(db.keys())
            if self._keys - in_file:
                CODELOG.debug(f"Shelf keys not found {path}")
                return dict(expected=self._keys, found=in_file)

    def mock(self, execution_context):
        path = self.path(execution_context)
        with shelve.open(str(path)) as db:
            CODELOG.info(f"mocking shelf with keys {self._keys}")
            for key in self._keys:
                db[key] = "marker"

    def remove(self, execution_context):
        path = self.path(execution_context)
        base = path.parent
        for dbm_file in base.glob(f"{path.name}.*"):
            dbm_file.unlink()
