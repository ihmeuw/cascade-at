import numpy as np
import pandas as pd

import pytest

from sqlalchemy.exc import StatementError

from cascade.dismod.db_wrapper import DismodFile, _get_engine


@pytest.fixture
def engine():
    return _get_engine(None)


@pytest.fixture
def base_file(engine):
    dm_file = DismodFile(engine, {"howdy": float}, {"there": int})
    dm_file.make_densities()
    ages = pd.DataFrame({"age": np.array([6.0, 22.0, 48.0])})
    dm_file.age = ages
    dm_file.time = pd.DataFrame({"time": [1997.0, 2005.0, 2017.0]})
    dm_file.integrand = pd.DataFrame({"integrand_name": ["prevalence"]})

    return dm_file


def test_wrong_type(base_file):
    ages = pd.DataFrame({"age": np.array(["strings", "for", "ages"])})
    base_file.age = ages
    with pytest.raises(StatementError):
        base_file.flush()


def test_is_dirty__initially(base_file):
    assert base_file._is_dirty("age")


def test_is_dirty__after_flush(base_file):
    base_file.flush()
    assert not base_file._is_dirty("age")


def test_is_dirty__after_modification(base_file):
    base_file.flush()
    base_file.age.loc[0, "age"] *= 2
    assert base_file._is_dirty("age")


def test_is_dirty__on_read(base_file, engine):
    base_file.flush()

    dm_file2 = DismodFile(engine, {"howdy": float}, {"there": int})

    dm_file2.age

    assert not dm_file2._is_dirty("age")


def test_is_dirty__not_yet_read(base_file):
    assert not base_file._is_dirty("foo_bar")


def test_dmfile_read(base_file, engine):
    ages = base_file.age
    times = base_file.time
    base_file.flush()

    dm_file2 = DismodFile(engine, {"howdy": float}, {"there": int})
    assert ages.equals(dm_file2.age)
    assert times.equals(dm_file2.time)


def test_reading_modified_columns(base_file, engine):
    base_file.flush()
    base_file.age.loc[0, "age"] *= 2
    ages = base_file.age.copy()
    base_file.flush()

    dm_file2 = DismodFile(engine, {"howdy": float}, {"there": int})
    assert ages.equals(dm_file2.age)
