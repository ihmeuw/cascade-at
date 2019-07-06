from pathlib import Path
import shelve
from types import SimpleNamespace
import pandas as pd
import sqlite3

from cascade.executor.execution_context import make_execution_context
from cascade.runner.job_graph import Job, RecipeIdentifier


class FileEntity:
    def __init__(self, relative_path, location_id=None):
        # If location_id isn't specified, it's the same location as the reader.
        self.relative_path = Path(relative_path)
        self.location_id = location_id

    def path(self, execution_context):
        db_path = execution_context.db_path(self.location_id) / self.relative_path
        return db_path


class DbFile(FileEntity):
    def __init__(self, relative_path, location_id=None, required_tables=None):
        super().__init__(relative_path, location_id)
        self._tables = set(required_tables)

    def validate(self, execution_context):
        path = self.path(execution_context)
        with sqlite3.connect(path) as conn:
            result = conn.execute("select name from sqlite_master where type='table'")
            tables = {x[0] for x in result}
            assert tables == self._tables


class PandasFile(FileEntity):
    def __init__(self, relative_path, location_id=None, required_columns=None):
        super().__init__(relative_path, location_id)
        self._columns = {key: set(cols) for (key, cols) in required_columns.items()}

    def validate(self, execution_context):
        path = self.path(execution_context)
        for key, cols in self._columns.items():
            df = pd.read_hdf(path, key=key)
            assert self._columns == set(df.columns)


class ShelfFile(FileEntity):
    def __init__(self, relative_path, location_id=None, required_keys=None):
        super().__init__(relative_path, location_id)
        self._keys = set(required_keys)

    def validate(self, execution_context):
        path = self.path(execution_context)
        with shelve.open(path) as db:
            assert self._keys == set(db.keys())


class FitFixed(Job):
    def __init__(self, recipe_id, local_settings):
        super().__init__("fit_fixed", recipe_id, local_settings)

        self.inputs = dict(
            db_file=DbFile("fixed.db", 23),
            parent_draws=PandasFile("posterior_predictions.hdf"),
            shelf=ShelfFile("shelf.shelf")
        )
        self.outputs = dict(
            fit_fixed=FileEntity("fixed.db")
        )

    def __call__(self, execution_context):
        covariate_data_spec = self.inputs["shelf"]
        covariate_data_spec.open(self.inputs["shelf"].path(execution_context))


def test_job_read_file():
    recipe_id = RecipeIdentifier(21, "estimate_location", "both")
    local_settings = SimpleNamespace()
    job = FitFixed(recipe_id, local_settings)
    assert job is not None
    ec = make_execution_context()
    assert ec is not None
