from cascade.core.context import ExecutionContext


def test_make_execution_context():
    ec = ExecutionContext()
    ec.parameters = {"database": "dismod-at-dev", "bundle_database": "epi"}
    assert ec.parameters.database == "dismod-at-dev"


def test_execution_context_file_interface(tmp_path):
    ec = ExecutionContext()
    ec.parameters = {"database": "dismod-at-dev"}
    ec.parameters.update(dict(a=47))
    assert ec.parameters.a == 47
    fit_db = ec.fit_db_path()
    assert fit_db.parent.exists()
