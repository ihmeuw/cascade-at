import sqlite3
from pathlib import Path

import gridengineapp

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class FileEntity(gridengineapp.FileEntity):
    """Responsible for making a path that is writable for a file.

    Args:
        relative_path (Path|str): Path to the file, relative to whatever
            directory the execution context says this ``location_id``
            should use.
        location_id (int): The location for which this file is written.
        sex (str): One of male, female, both.
    """
    def __init__(self, execution_context, relative_path, location_id, sex=None):
        # If location_id isn't specified, it's the same location as the reader.
        base_directory = execution_context.model_base_directory(location_id, sex)
        full_path = base_directory / Path(relative_path)
        super().__init__(full_path)


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
    def __init__(self, execution_context, relative_path, location_id=None, sex=None, required_tables=None):
        super().__init__(execution_context, relative_path, location_id, sex)
        self._tables = set(required_tables) if required_tables else set()

    def validate(self):
        """Validate by checking which tables exist because this establishes
        which steps have been run. Doesn't look inside the tables.

        Returns:
            None, on success, or a string on error.
        """
        super_message = super().validate()
        if super_message is not None:
            return super_message
        if not self._tables:
            return
        with sqlite3.connect(str(self.path)) as conn:
            result = conn.execute("select name from sqlite_master where type='table'")
            tables = {x[0] for x in result}

        if self._tables - tables:
            return f"found {tables}, expected {self._tables}"

    def mock(self):
        """Creates a sqlite3 file with the given tables. They don't correspond
        to the columns that are actually expected."""
        if not self._tables:
            self.path.open("w").close()
        with sqlite3.connect(str(self.path)) as conn:
            for table in self._tables:
                conn.execute(f"CREATE TABLE {table} (key text, value text)")
            conn.commit()


class PandasFile(gridengineapp.PandasFile):
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
    def __init__(self, execution_context, relative_path, location_id=None, sex=None, required_frames=None):
        base_directory = execution_context.model_base_directory(location_id, sex)
        full_path = base_directory / Path(relative_path)
        super().__init__(full_path, required_frames)


class ShelfFile(gridengineapp.ShelfFile):
    """Responsible for validating a Python shelf file.

    Args:
        relative_path (Path|str): Path to the file, relative to whatever
            directory the execution context says this ``location_id``
            should use.
        location_id (int): The location for which this file is written.
        sex (str): One of male, female, both.
        required_keys (Set[str]): String names of variables to find in the file.
    """
    def __init__(self, execution_context, relative_path, location_id=None, sex=None, required_keys=None):
        base_directory = execution_context.model_base_directory(location_id, sex)
        full_path = base_directory / Path(relative_path)
        super().__init__(full_path, required_keys)
