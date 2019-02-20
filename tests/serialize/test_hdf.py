from h5py import File
from numpy import isclose

from cascade.model.var import Var
from cascade.model.dismod_groups import DismodGroups
from cascade.serialize.hdf import write_var, read_var, write_group, read_group


def test_write_vars_one_field(tmp_path):
    id_draw = Var([0, 50, 100], [1995, 2015])
    for a, t in id_draw.age_time():
        id_draw[a, t] = a * 2 + t

    with File(str(tmp_path / "test.hdf5"), "w") as f:
        g = f.create_group("var1")
        written_data = write_var(g, id_draw, "iota")
        assert written_data.shape == (3, 2)

    with File(str(tmp_path / "test.hdf5"), "r") as r:
        g = r["var1"]
        v = read_var(g["iota"])
        for a, t in id_draw.age_time():
            assert isclose(v[a, t], id_draw[a, t])


def new_var(idx):
    id_draw = Var([0, 50 + idx, 100], [1995, 2015])
    for a, t in id_draw.age_time():
        id_draw[a, t] = a * idx + t
    return id_draw


def test_write_groups(tmp_path):
    dg = DismodGroups()
    dg.rate["iota"] = new_var(1)
    dg.rate["omega"] = new_var(2)
    dg.random_effect[("iota", None)] = new_var(3)
    dg.random_effect[("omega", 1)] = new_var(4)
    dg.random_effect[("omega", 2)] = new_var(5)
    dg.alpha[("iota", "traffic")] = new_var(6)

    with File(str(tmp_path / "test.hdf5"), "w") as f:
        g = f.create_group("var1")
        write_group(g, dg)

    with File(str(tmp_path / "test.hdf5"), "r") as r:
        g = r["var1"]
        rg = read_group(g)

    assert "iota" in rg.rate
    assert "omega" in rg.rate
    assert ("iota", None) in rg.random_effect
    assert ("omega", 1) in rg.random_effect
    assert ("omega", 2) in rg.random_effect
    assert ("iota", "traffic") in dg.alpha
    assert not dg.beta
    assert not dg.gamma

    assert rg.rate["iota"] == dg.rate["iota"]
    assert rg.random_effect[("omega", 1)] == dg.random_effect[("omega", 1)]
