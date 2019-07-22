from os import walk

import pytest
from gridengineapp import entry
from numpy.random import RandomState

from cascade.executor.create_settings import SettingsChoices, create_settings
from cascade.executor.dismodel_main import DismodAT
from cascade.executor.execution_context import make_execution_context
from cascade.input_data.db.locations import location_hierarchy


@pytest.mark.parametrize("meid,mvid", [
    (None, 268613),
])
def test_with_known_id(ihme, meid, mvid, tmp_path):
    """This runs the equivalent of dismodel_main.main"""
    # no-upload keeps this from going to the databases when it's done.
    args = ["--no-upload", "--db-only", "--skip-cache",
            "--base-directory", str(tmp_path)]
    if mvid:
        args += ["--mvid", str(mvid)]
    elif meid:
        args += ["--meid", str(meid)]
    else:
        assert meid or mvid

    app = DismodAT()
    entry(app, args)


@pytest.mark.parametrize("draw", list(range(1)))
def test_retrieve_data(ihme, draw, tmp_path):
    ec = make_execution_context()
    rng = RandomState(524287 + 131071 * draw)
    locs = location_hierarchy(5, 429)

    choices = SettingsChoices(rng, None)
    settings = create_settings(choices, locs)

    app = DismodAT(locs, settings, ec)
    arg_list = [
        "--no-upload", "--db-only",
        "--base-directory", str(tmp_path),
        "--location", "0",
        "--recipe", "bundle_setup",  # We are asking for one particular recipe.
    ]
    entry(app, arg_list)

    for dirpath, dirnames, filenames in walk(tmp_path):
        if filenames:
            print(f"{dirpath}:")
            print(f"\t{filenames}")

    meid = "23514"
    mvid = "267890"
    loc = "0"
    sex = "both"
    base = tmp_path / meid / mvid / "0" / loc / sex
    assert (base / "globaldata.hdf").exists()
    assert len(list(base.glob("globalvars*"))) > 0
