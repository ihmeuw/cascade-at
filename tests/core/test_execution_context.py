from pathlib import Path

import pytest

from cascade.core.context import ExecutionContext


def test_make_execution_context():
    ec = ExecutionContext()
    ec.parameters = {"database": "dismod-at-dev", "bundle_database": "epi"}
    assert ec.parameters.database == "dismod-at-dev"
    assert hasattr(ec.parameters, "database")


@pytest.mark.parametrize("meid,mvid,expected", [
    (4321, 267737, "./4321/267737/1/120"),
    (None, 267737, "./mvid/267737/1/120"),
])
def test_execution_context_file_interface(meid, mvid, expected):
    ec = ExecutionContext()
    ec.parameters = {"database": "dismod-at-dev"}
    ec.parameters.base_directory = "."
    ec.parameters.organizational_mode = "infrastructure"
    assert ec.parameters.base_directory == "."
    ec.parameters.modelable_entity_id = meid
    ec.parameters.model_version_id = mvid
    db_path = ec.db_path(120)
    expect_base = (Path(".") / expected).expanduser()
    assert db_path == expect_base


@pytest.mark.parametrize("meid,mvid,expected", [
    (4321, 267737, "mytmp/120"),
    (None, 267737, "mytmp/120"),
])
def test_execution_context_local_file_interface(meid, mvid, expected):
    ec = ExecutionContext()
    ec.parameters = {"database": "dismod-at-dev"}
    ec.parameters.base_directory = "./mytmp"
    ec.parameters.organizational_mode = "local"
    ec.parameters.modelable_entity_id = meid
    ec.parameters.model_version_id = mvid
    db_path = ec.db_path(120)
    expect_base = (Path(".") / expected).expanduser()
    assert db_path == expect_base
