from cascade_at.context.model_context import Context


def test_context_files():
    c = Context(0, 'dismod-at-dev', make=False, configure_application=False)
    assert str(c.inputs_dir) == 'cascade_dir/data/0/inputs'
    assert str(c.outputs_dir) == 'cascade_dir/data/0/outputs'
    assert str(c.database_dir) == 'cascade_dir/data/0/dbs'


def test_context_location_sex():
    c = Context(0, 'dismod-at-dev', make=False, configure_application=False)
    assert str(c.db_file(1, 3, make=False)) == 'cascade_dir/data/0/dbs/1/3/dismod.db'

