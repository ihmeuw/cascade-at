import shutil
from os import walk
from pathlib import Path

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
        "--location", "0", "--skip-cache",
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


@pytest.fixture
def globaldir(request, tmp_path):
    """This is a cached store of the results of bundle setup because
    bundle setup is so slow downloading from the database.

    Does this look like a pain in the butt? Yes. But it's sometimes five
    minutes, and this sets us up to test any part of the hierarchy.
    Use ``pytest --clear-cache`` to erase this entry and re-download.
    """
    def by_draw(draw):
        ec = make_execution_context()
        rng = RandomState(524287 + 131071 * draw)
        locs = location_hierarchy(6, 429)

        choices = SettingsChoices(rng, None)
        settings = create_settings(choices, locs)

        base_path = request.config.cache.get(f"run_global/globaldir{draw}", None)
        if isinstance(base_path, str):
            base_path = Path(base_path)
        if base_path is None or not (base_path / "globaldata.hdf").exists():
            app = DismodAT(locs, settings, ec)
            # skip-cache says to use Tier 2 data.
            arg_list = [
                "--no-upload", "--db-only", "-v",
                "--base-directory", str(tmp_path),
                "--location", "0", "--skip-cache",
                "--recipe", "bundle_setup",  # We are asking for one particular recipe.
            ]
            entry(app, arg_list)
            meid = "23514"
            mvid = "267890"
            loc = "0"
            sex = "both"
            base = tmp_path / meid / mvid / "0" / loc / sex
            global_data = base / "globaldata.hdf"
            assert global_data.exists()
            request.config.cache.set(f"run_global/globaldir{draw}", str(base))
            base_path = base
        return settings, base_path
    return by_draw


@pytest.mark.parametrize("draw", list(range(1)))
def test_run_global(ihme, draw, tmp_path, globaldir):
    settings, global_data = globaldir(draw)
    ec = make_execution_context()
    locs = location_hierarchy(6, 429)
    app = DismodAT(locs, settings, ec)
    # We copy a previously-computed directory into the correct location
    # in tmp_path because it takes so long to download the data that
    # testing is a bear.
    meid = "23514"
    mvid = "267890"
    loc = "0"
    sex = "both"
    base = tmp_path / meid / mvid / "0" / loc / sex
    try:
        shutil.copytree(global_data, base)
    except FileExistsError:
        assert (base / "globaldata.hdf").exists()

    print(f"Retrieved data, now running.")
    arg_list = [
        "--verbose-app", "-v",
        "--no-upload", "--db-only",
        "--base-directory", str(tmp_path),
        "--location", "32",
        "--recipe", "estimate_location",  # We are asking for one particular recipe.
    ]
    entry(app, arg_list)

    # For debugging, prints all files that were just made under tmp_path.
    for dirpath, dirnames, filenames in walk(tmp_path):
        if filenames:
            print(f"{dirpath}:")
            print(f"\t{filenames}")

    meid = "23514"
    mvid = "267890"
    loc = "32"
    sex = "female"
    base = tmp_path / meid / mvid / "0" / loc / sex
    assert (base / "fit.db").exists()
