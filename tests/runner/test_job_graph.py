from pathlib import Path
from types import SimpleNamespace

from cascade.executor.execution_context import make_execution_context
from cascade.runner.job_graph import Job, RecipeIdentifier


class FileEntity:
    def __init__(self, relative_path, location_id=None):
        # If location_id isn't specified, it's the same location as the reader.
        self.location_id = location_id
        self.relative_path = Path(relative_path)

    def path(self, execution_context):
        db_path = execution_context.db_path(self.location_id) / self.relative_path
        return db_path


class FitFixed(Job):
    def __init__(self, recipe_id, local_settings):
        super().__init__("fit_fixed", recipe_id, local_settings)

        self.inputs = dict(
            db_file=FileEntity("fixed.db", 23),
            parent_draws=FileEntity("posterior_predictions.csv"),
        )
        self.outputs = dict(
            fit_fixed=FileEntity("fixed.db")
        )

    def __call__(self, execution_context):
        pass


def test_job_read_file():
    recipe_id = RecipeIdentifier(21, "estimate_location", "both")
    local_settings = SimpleNamespace()
    job = FitFixed(recipe_id, local_settings)
    assert job is not None
    ec = make_execution_context()
    assert ec is not None
