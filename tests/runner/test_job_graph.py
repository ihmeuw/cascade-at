from types import SimpleNamespace

from cascade.executor.execution_context import make_execution_context
from cascade.runner.data_passing import FileEntity
from cascade.runner.job_graph import CascadeJob, RecipeIdentifier


class FitFixed(CascadeJob):
    def __init__(self, recipe_id, local_settings, execution_context):
        super().__init__("fit_fixed", recipe_id, local_settings, execution_context)

        self.inputs.update(dict(
            db_file=FileEntity(execution_context, "fixed.db", 23),
            parent_draws=FileEntity(execution_context, "posterior_predictions.csv", 23),
        ))
        self.outputs.update(dict(
            fit_fixed=FileEntity(execution_context, "fixed.db", 47)
        ))

    def __call__(self):
        raise RuntimeError(f"Don't call me.")


def test_job_read_file():
    ec = make_execution_context()
    recipe_id = RecipeIdentifier(21, "estimate_location", "both")
    local_settings = SimpleNamespace()
    job = FitFixed(recipe_id, local_settings, ec)
    assert job is not None


def test_mock_job():
    execution_context = make_execution_context(
        gbd_round_id=6, num_processes=4
    )
    recipe_id = RecipeIdentifier(21, "estimate_location", "both")
    local_settings = SimpleNamespace()
    job = FitFixed(recipe_id, local_settings, execution_context)
    job.mock_run()
