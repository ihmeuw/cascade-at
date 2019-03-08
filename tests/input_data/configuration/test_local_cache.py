from cascade.input_data.configuration.local_cache import LocalCache


def test_holds_two():
    lc = LocalCache(maxsize=2)
    lc.set("one", 3)
    assert lc.get("one") == 3
    lc.set("two", 4)
    assert lc.get("one") == 3
    assert lc.get("two") == 4
    lc.set("three", 7)
    assert not lc.get("one")
    assert lc.get("two") == 4
    assert lc.get("three") == 7
