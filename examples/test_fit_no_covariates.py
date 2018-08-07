from argparse import Namespace
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sqlite3

import cascade.input_data.db.bundle
from cascade.dismod.db.wrapper import DismodFile, _get_engine
import fit_no_covariates as example


def test_bundle_to_integrand():
    df = pd.DataFrame(dict(
        measure=["Sincidence", "mtexcess"],
        mean=[0.001, 0.000],
        sex=["Male", "Female"],
        standard_error=["0.000", "0.000"],
        age_start=[0.0, 5.0],
        age_end=[5.0, 15.0],
        year_start=[1965, 1970],
        year_end=[1965, 1970],
    ))
    config = Namespace(location_id=26)
    integrand = example.bundle_to_observations(config, df)
    final_columns = set(("measure density location_id weight age_start "
                         "age_end year_start year_end mean standard_error").split())
    assert set(integrand.columns) == final_columns
    assert integrand["year_start"].dtype == np.float
    assert integrand["year_end"].dtype == np.float
    return integrand


def test_bundle_cache(monkeypatch, tmpdir):

    def mock_bundle(context, bundle_id, tier):
        return (pd.DataFrame(dict(
            measure=["incidence", "mtexcess"],
            age_start=[0.00, 5.0],
            age_end=[5.0, 10.0],
            year_start=[1965, 1970],
            year_end=[1970, 1970],
            mean=[0.00, 0.005],
            standard_error=[0.000, 0.01],
        )), pd.DataFrame())

    monkeypatch.setattr(cascade.input_data.db.bundle, "bundle_with_study_covariates",
                        mock_bundle)
    current_working = os.getcwd()
    os.chdir(tmpdir)

    bundle, covariates = example.cached_bundle_load(None, 1234, 2)
    assert "age_start" in bundle.columns
    assert Path("1234.pkl").exists()

    def mock_fail(context, bundle_id, tier):
        raise RuntimeError("should not be called")

    monkeypatch.setattr(cascade.input_data.db.bundle, "bundle_with_study_covariates",
                        mock_fail)
    bundle, covariates = example.cached_bundle_load(None, 1234, 2)
    assert "age_start" in bundle.columns

    os.chdir(current_working)


def test_retrieve_external_data():
    config = Namespace()
    config.bundle_id = 3209
    config.tier_idx = 2
    config.location_id=26
    config.options = dict(
        non_zero_rates = "iota rho chi omega"
    )

    # Get the bundle and process it.
    inputs = example.retrieve_external_data(config)


logging.basicConfig(level=logging.DEBUG)


def test_write_own_id(tmpdir):
    fname = Path(f"{tmpdir}/z.db")
    bundle_file_engine = _get_engine(fname)
    bundle_fit = DismodFile(bundle_file_engine, {}, {})
    node = pd.DataFrame({
        "node_name": ["samoa", "china", "us"],
        "parent": np.array([102, None, None], dtype=np.float),
        },
        index=[26, 6, 102],
    )
    print(f"indexed node is {node}")
    bundle_fit.node = node
    print(f"parent type in {bundle_fit.node['parent'].dtype}")
    bundle_fit.flush()

    # Need to use raw sqlite3 to see what's really in the database.
    conn = sqlite3.connect(str(fname))
    results = conn.execute("SELECT * from node").fetchall()

    print(results)
    for index, name, value in results:
        assert isinstance(index, int)
        assert isinstance(name, str)
        assert isinstance(value, int) or value is None
