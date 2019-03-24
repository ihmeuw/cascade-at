import os

import pytest

from cascade.executor.execution_context import make_execution_context
from cascade.input_data.db.configuration import (
    trim_config, DO_REMOVE, load_raw_settings_meid, load_raw_settings_mvid,
    load_raw_settings_file
)


def test_trim_config__trivially_empty():
    assert trim_config({}) is DO_REMOVE
    assert trim_config([]) is DO_REMOVE
    assert trim_config(None) is DO_REMOVE
    assert trim_config("") is DO_REMOVE


def test_trim_config__trivially_not_empty():
    assert trim_config("test") == "test"
    assert trim_config(["test"]) == ["test"]
    assert trim_config({"test": "test", "another_test": "another_test"}) == {
        "test": "test",
        "another_test": "another_test",
    }


def test_trim_config__nested():
    assert trim_config({"thing": 10, "other": {"inner": 100, "more_inner": [1, 2, 4]}}) == {
        "thing": 10,
        "other": {"inner": 100, "more_inner": [1, 2, 4]},
    }
    assert trim_config({"thing": 10, "other": {}}) == {"thing": 10}
    assert trim_config({"thing": 10, "other": {"bleh": [], "blah": {}}}) == {"thing": 10}


def test_load_settings_meid(ihme):
    ec = make_execution_context()
    config, mvid = load_raw_settings_meid(ec, 1989)
    assert isinstance(config, dict)
    assert len(config) > 0
    assert isinstance(mvid, int)
    assert mvid > 0


def test_load_settings_meid_bad(ihme):
    ec = make_execution_context()
    bad_meid = "198919891989"
    with pytest.raises(RuntimeError) as re:
        load_raw_settings_meid(ec, bad_meid)
    assert bad_meid in str(re)


def test_load_settings_meid_string(ihme):
    ec = make_execution_context()
    config, mvid = load_raw_settings_meid(ec, "1989")
    assert isinstance(config, dict)
    assert len(config) > 0
    assert isinstance(mvid, int)
    assert mvid > 0


def test_load_settings_mvid(ihme):
    ec = make_execution_context()
    config1, mvid1 = load_raw_settings_meid(ec, 1989)
    assert isinstance(config1, dict)
    config, mvid = load_raw_settings_mvid(ec, mvid1)
    assert isinstance(config, dict)
    assert len(config) > 0
    assert isinstance(mvid, int)
    assert mvid > 0


def test_load_settings_mvid_str(ihme):
    ec = make_execution_context()
    config1, mvid1 = load_raw_settings_meid(ec, 1989)
    assert isinstance(config1, dict)
    config, mvid = load_raw_settings_mvid(ec, str(mvid1))
    assert isinstance(config, dict)
    assert len(config) > 0
    # Put a string in, get a string out.
    assert isinstance(mvid, str)


def test_load_settings_file(tmpdir, ihme):
    f = os.path.join(tmpdir, "unit_test.json")
    with open(f, "w") as test_json:
        test_json.write('{"model": {"modelable_entity_id": 1989}}')

    ec = make_execution_context()
    config, mvid = load_raw_settings_file(ec, f)
    assert isinstance(config, dict)
    assert len(config) > 0
    assert isinstance(mvid, int)
    assert mvid > 0
