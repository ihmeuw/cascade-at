from types import SimpleNamespace

from cascade.executor.execution_context import make_execution_context
from cascade.runner.data_passing import FileEntity
from cascade.runner.job_graph import CascadeJob, RecipeIdentifier


class FitFixed(CascadeJob):
    def __init__(self, recipe_id, local_settings):
        super().__init__("fit_fixed", recipe_id, local_settings)

        self.inputs = dict(
            db_file=FileEntity("fixed.db", 23),
            parent_draws=FileEntity("posterior_predictions.csv", 23),
        )
        self.outputs = dict(
            fit_fixed=FileEntity("fixed.db", 47)
        )

    def __call__(self, execution_context):
        raise RuntimeError(f"Don't call me.")


def test_job_read_file():
    recipe_id = RecipeIdentifier(21, "estimate_location", "both")
    local_settings = SimpleNamespace()
    job = FitFixed(recipe_id, local_settings)
    assert job is not None
    ec = make_execution_context()
    assert ec is not None


def test_mock_job():
    recipe_id = RecipeIdentifier(21, "estimate_location", "both")
    local_settings = SimpleNamespace()
    job = FitFixed(recipe_id, local_settings)
    ec = make_execution_context()
    job.mock_run(ec, check_inputs=False)
