import enum

import numpy as np
import pandas as pd

import pytest

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Enum

from cascade.dismod.db.wrapper import DismodFile, _get_engine, _validate_data
from cascade.dismod.db import DismodFileError


@pytest.fixture
def engine():
    return _get_engine(None)


@pytest.fixture
def base_file(engine):
    dm_file = DismodFile(engine, {"howdy": float}, {"there": int})
    dm_file.make_densities()
    ages = pd.DataFrame({"age": np.array([6.0, 22.0, 48.0])})
    ages["age_id"] = ages.index
    dm_file.age = ages
    dm_file.time = pd.DataFrame({"time": [1997.0, 2005.0, 2017.0]})
    dm_file.time["time_id"] = dm_file.time.index
    dm_file.integrand = pd.DataFrame({"integrand_name": ["prevalence"]})
    dm_file.integrand["integrand_id"] = dm_file.integrand.index

    return dm_file


# skip this until Drew's version of the db serialization process is merged in
# def test_wrong_type(base_file):
#    ages = pd.DataFrame({"age": np.array(["strings", "for", "ages"])})
#    ages["age_id"] = ages.index
#    base_file.age = ages
#    with pytest.raises(DismodFileError):
#        base_file.flush()


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
    assert ages.sort_index(axis=1).equals(dm_file2.age.sort_index(axis=1))
    assert times.sort_index(axis=1).equals(dm_file2.time.sort_index(axis=1))


def test_reading_modified_columns(base_file, engine):
    base_file.flush()
    base_file.age.loc[0, "age"] *= 2
    ages = base_file.age.copy()
    base_file.flush()

    dm_file2 = DismodFile(engine, {"howdy": float}, {"there": int})
    assert ages.sort_index(axis=1).equals(dm_file2.age.sort_index(axis=1))


DummyBase = declarative_base()


class DummyTable(DummyBase):
    __tablename__ = "test_table"
    primary_key_column = Column(Integer(), primary_key=True)
    integer_column = Column(Integer())
    float_column = Column(Float())
    nonnullable_column = Column(Integer(), nullable=False)
    string_column = Column(String())
    enum_column = Column(Enum(enum.Enum("Bee", "bumble honey carpenter wool_carder")))


def test_validate_data__happy_path():
    data = pd.DataFrame(
        {
            "integer_column": [1, 2, 3],
            "float_column": [1.0, 2.0, 3.0],
            "string_column": ["a", "b", "c"],
            "enum_column": ["bumble", "honey", "carpenter"],
            "nonnullable_column": [1, 2, 3],
        }
    )
    _validate_data(DummyTable.__table__, data)


def test_validate_data__bad_integer():
    data = pd.DataFrame({"integer_column": [1.0, 2.0, 3.0], "nonnullable_column": [1, 2, 3]})
    with pytest.raises(DismodFileError) as excinfo:
        _validate_data(DummyTable.__table__, data)

    assert "integer_column" in str(excinfo.value)


def test_validate_data__bad_float():
    data = pd.DataFrame({"float_column": ["1.0", "2.0", "3.0"], "nonnullable_column": [1, 2, 3]})
    with pytest.raises(DismodFileError) as excinfo:
        _validate_data(DummyTable.__table__, data)

    assert "float_column" in str(excinfo.value)


def test_validate_data__bad_string():
    data = pd.DataFrame({"string_column": [1, 2, 3], "nonnullable_column": [1, 2, 3]})
    with pytest.raises(DismodFileError) as excinfo:
        _validate_data(DummyTable.__table__, data)
    assert "string_column" in str(excinfo.value)


def test_validate_data__bad_enum():
    data = pd.DataFrame({"enum_column": [1, 2, 3], "nonnullable_column": [1, 2, 3]})
    with pytest.raises(DismodFileError) as excinfo:
        _validate_data(DummyTable.__table__, data)
    assert "enum_column" in str(excinfo.value)


def test_validate_data__extra_column():
    data = pd.DataFrame({"nonnullable_column": [1, 2, 3], "other_column": [1, 2, 3]})
    with pytest.raises(DismodFileError) as excinfo:
        _validate_data(DummyTable.__table__, data)
    assert "other_column" in str(excinfo.value)


def test_validate_data__missing_column():
    data = pd.DataFrame({"integer_column": [1, 2, 3]})
    with pytest.raises(DismodFileError) as excinfo:
        _validate_data(DummyTable.__table__, data)
    assert "nonnullable_column" in str(excinfo.value)
