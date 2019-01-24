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
