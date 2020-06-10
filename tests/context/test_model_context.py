import pytest

from cascade_at.context.model_context import Context


@pytest.fixture
def context(tmp_path):
    c = Context(model_version_id=0, make=True, configure_application=False,
                root_directory=tmp_path)
    return c


def test_context_files(context):
    assert str(context.inputs_dir).endswith('cascade_dir/data/0/inputs')
    assert str(context.outputs_dir).endswith('cascade_dir/data/0/outputs')
    assert str(context.database_dir).endswith('cascade_dir/data/0/dbs')


def test_context_location_sex(context):
    assert str(context.db_file(1, 3)).endswith('cascade_dir/data/0/dbs/1/3/dismod.db')
