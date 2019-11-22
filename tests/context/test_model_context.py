from cascade_at.context.model_context import Context


def test_context_files():
    c = Context(0, 'dismod-at-dev', make=False, cascade_dir='/tmp')
    assert str(c.inputs_dir.absolute()) == '/tmp/data/0/inputs'
    assert str(c.outputs_dir.absolute()) == '/tmp/data/0/outputs'
    assert str(c.database_dir.absolute()) == '/tmp/data/0/dbs'


def test_context_location_sex():
    c = Context(0, 'dismod-at-dev', make=False, cascade_dir='/tmp')
    assert str(c.db_folder(1, 3, make=False)) == '/tmp/data/0/dbs/1/3'
