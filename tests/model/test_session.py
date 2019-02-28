import logging
from math import nan
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from cascade.model import Session, Var, DismodGroups
from cascade.model.session import _run_with_async_logging


def test_options(dismod):
    locations = pd.DataFrame(dict(
        name=["global"],
        parent_id=[nan],
        location_id=[1],
    ))
    parent_location = 1
    db_file = Path("option.db")
    session = Session(locations, parent_location, db_file)
    opts = dict(
        meas_std_effect="add_std_scale_all",
        parent_node_name="global",
        quasi_fixed="false",
        derivative_test_fixed="second-order",
        max_num_iter_fixed=100,
        print_level_fixed=0,
        tolerance_fixed=1e-10,
        age_avg_split=[0.19, 0.5, 1, 2],
    )
    session.set_option(**opts)

    iota = Var([20], [2000])
    iota.grid.loc[:, "mean"] = 0.01
    model_var = DismodGroups()
    model_var.rate["iota"] = iota
    avgints = pd.DataFrame(dict(
        integrand="susceptible",
        location=parent_location,
        age_lower=np.linspace(0, 120, 2),
        age_upper=np.linspace(0, 120, 2),
        time_lower=2000,
        time_upper=2000,
    ))
    # Run a predict in order to verify the options are accepted.
    session.predict(model_var, avgints, parent_location)


def test_run_with_async_logging__stdout(caplog):
    with caplog.at_level(logging.INFO):
        exit_code, stdout, stderr = _run_with_async_logging(["echo", "stdout test"])
    assert exit_code == 0
    assert stdout == "stdout test\n"
    assert stderr == ""
    assert "stdout test" in caplog.text


def test_run_with_async_logging__stderr(caplog):
    exit_code, stdout, stderr = _run_with_async_logging(["bash", "-c", "echo stderr test 1>&2"])
    assert exit_code == 0
    assert stdout == ""
    assert stderr == "stderr test\n"
    assert "stderr test" in caplog.text


def test_run_with_async_logging__non_zero_exit():
    exit_code, stdout, stderr = _run_with_async_logging(["false"])
    assert exit_code != 0
    assert stdout == ""
    assert stderr == ""


def test_run_with_async_logging__bad_executable():
    with pytest.raises(FileNotFoundError):
        _run_with_async_logging(["blargh"])
