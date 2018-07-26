import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import StatementError

from cascade.dismod.db_wrapper import DismodFile, _get_engine


def test_add_age():
    engine = _get_engine(None)
    dm_file = DismodFile(engine, {"howdy": float}, {"there": int})
    dm_file.make_densities()
    ages = pd.DataFrame({"age": np.array([6.0, 22.0, 48.0])})
    dm_file.age = ages
    dm_file.time = pd.DataFrame({"time": [1997, 2005, 2017]})
    dm_file.integrand = pd.DataFrame({"integrand_name": ["prevalence"]})

    dm_file.diagnostic_print()


def test_wrong_type():
    engine = _get_engine(None)
    dm_file = DismodFile(engine, {"howdy": float}, {"there": int})
    dm_file.make_densities()
    ages = pd.DataFrame({"age": np.array(["strings", "for", "ages"])})
    dm_file.age = ages
    with pytest.raises(StatementError):
        dm_file.flush()


def test_dmfile_read():
    engine = _get_engine(None)
    dm_file = DismodFile(engine, {"howdy": float}, {"there": int})
    dm_file.make_densities()
    ages = pd.DataFrame({"age": np.array([6.0, 22.0, 48.0])})
    dm_file.age = ages
    dm_file.time = pd.DataFrame({"time": [1997, 2005, 2017]})
    dm_file.integrand = pd.DataFrame({"integrand_name": ["prevalence"]})

    time_df = dm_file.time
    assert time_df.loc[0, "time"] == 1997
