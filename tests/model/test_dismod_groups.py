import pytest

from cascade.model import DismodGroups


def test_create():
    dg = DismodGroups()
    dg.rate["iota"] = 7
    assert dg.rate["iota"] == 7
    assert dg["rate"]["iota"] == 7

    dg.random_effect[("traffic", "omega")] = 42
    assert dg.random_effect[("traffic", "omega")] == 42
    assert dg["random_effect"][("traffic", "omega")] == 42

    dg["alpha"][("traffic", "rho")] = 63
    assert dg.alpha[("traffic", "rho")] == 63
    assert dg["alpha"][("traffic", "rho")] == 63

    assert id(dg.data["rate"]) == id(dg.rate)


def test_no_do():
    """Demonstrates you can't assign the wrong way."""
    dg = DismodGroups()
    # Would assign to a nonexistent group.
    with pytest.raises(AttributeError):
        dg.rates["iota"] = 7

    # Would overwrite a whole group, creating a shadow dictionary.
    with pytest.raises(AttributeError):
        dg.random_effect = dict()

    # But you can still assign other things (in case of subclassing)
    dg.foo = "bar"


def test_misaligned():
    dg0 = DismodGroups()
    dg0.rate["iota"] = 7
    dg0.gamma[("traffic", 2)] = 9

    dg1 = DismodGroups()
    dg1.rate["iota"] = 7
    dg1.gamma[("traffic", 2)] = 9

    left = dg0.check_alignment(dg1)
    assert not left

    dg1.alpha[("sdi", 7)] = 42
    right = dg0.check_alignment(dg1)
    assert right is not None


def test_aligned_none():
    dg0 = DismodGroups()
    dg0.rate["iota"] = 7
    dg0.random_effect[("iota", 2)] = 9
    dg0.random_effect[("iota", 3)] = 9

    dg1 = DismodGroups()
    dg1.rate["iota"] = 7
    dg1.random_effect[("iota", None)] = 9

    left = dg0.check_alignment(dg1)
    assert not left


def test_aligned_more_than_one():
    dg0 = DismodGroups()
    dg0.rate["iota"] = 7
    dg0.random_effect[("iota", 2)] = 9
    dg0.random_effect[("iota", 3)] = 9
    dg0.random_effect[("rho", 2)] = 7
    dg0.random_effect[("rho", 3)] = 7
    dg0.random_effect[("omega", None)] = 14

    dg1 = DismodGroups()
    dg1.rate["iota"] = 7
    dg1.random_effect[("iota", None)] = 9
    dg1.random_effect[("rho", 2)] = 7
    dg1.random_effect[("rho", 3)] = 7
    dg1.random_effect[("omega", None)] = 9

    left = dg0.check_alignment(dg1)
    assert not left


def test_unaligned_more_than_one():
    dg0 = DismodGroups()
    dg0.rate["iota"] = 7
    dg0.random_effect[("iota", 2)] = 9
    dg0.random_effect[("iota", 3)] = 9
    dg0.random_effect[("rho", 2)] = 7
    dg0.random_effect[("rho", 3)] = 7
    dg0.random_effect[("omega", None)] = 14
    dg0.random_effect[("chi", None)] = 20

    dg1 = DismodGroups()
    dg1.rate["iota"] = 7
    dg1.random_effect[("iota", None)] = 9
    dg1.random_effect[("rho", 2)] = 7
    dg1.random_effect[("rho", 3)] = 7
    dg1.random_effect[("omega", None)] = 9

    left = dg0.check_alignment(dg1)
    assert left is not None


def test_unaligned_too_many():
    dg0 = DismodGroups()
    dg0.rate["iota"] = 7
    dg0.random_effect[("iota", 2)] = 9
    dg0.random_effect[("iota", 3)] = 9
    dg0.random_effect[("iota", 5)] = 9
    dg0.random_effect[("rho", 2)] = 7
    dg0.random_effect[("rho", 3)] = 7
    dg0.random_effect[("omega", None)] = 14
    dg0.random_effect[("chi", None)] = 20

    dg1 = DismodGroups()
    dg1.rate["iota"] = 7
    dg1.random_effect[("iota", None)] = 9
    dg1.random_effect[("rho", 2)] = 7
    dg1.random_effect[("rho", 3)] = 7
    dg1.random_effect[("omega", None)] = 9

    left = dg0.check_alignment(dg1)
    assert left is not None
