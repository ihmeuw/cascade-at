import numpy as np
import os
import pandas as pd
from pathlib import Path

import cascade.input_data.db.bundle
import fit_no_covariates as example


def test_bundle_to_integrand():
    df = pd.DataFrame(dict(
        measure=["incidence", "mtexcess"],
        mean=[0.001, 0.000],
        sex=["Male", "Female"],
        standard_error=["0.000", "0.000"],
        age_start=[0.0, 5.0],
        age_end=[5.0, 15.0],
        year_start=[1965, 1970],
        year_end=[1965, 1970],
    ))
    integrand = example.bundle_to_integrand(df, 26)
    final_columns = set("integrand age_lower age_upper time_lower time_upper meas_value meas_std hold_out".split())
    assert set(integrand.columns) == final_columns
    assert integrand["time_lower"].dtype == np.float
    assert integrand["time_upper"].dtype == np.float
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
