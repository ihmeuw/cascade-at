from cascade.input_data.db.configuration import trim_config, DO_REMOVE


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
