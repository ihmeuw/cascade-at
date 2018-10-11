import enum
import time

import numpy as np
import pandas as pd

import pytest

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Enum

from cascade.dismod.db.wrapper import DismodFile, _get_engine, _validate_data, _ordered_by_foreign_key_dependency
from cascade.dismod.db import DismodFileError
from cascade.dismod.db.metadata import Base as DismodFileBase


@pytest.fixture
def engine():
    return _get_engine(None)


@pytest.fixture
def base_file(engine):
    dm_file = DismodFile(engine)
    dm_file.make_densities()
    ages = pd.DataFrame({"age": np.array([6.0, 22.0, 48.0])})
    ages["age_id"] = ages.index
    dm_file.age = ages
    dm_file.time = pd.DataFrame({"time": [1997.0, 2005.0, 2017.0]})
    dm_file.integrand = pd.DataFrame({"integrand_name": ["prevalence"], "minimum_meas_cv": [0.0]})

    return dm_file


@pytest.mark.parametrize(
    "input,expected",
    [
        ("age time integrand density", "age density integrand time"),
        ("density prior age", "age density prior"),
        ("node", "node"),
        ("weight_grid age time weight", "age time weight weight_grid"),
        ("avgint node weight", "node weight avgint"),
    ],
)
def test_ordering_of_tables(input, expected):
    out = list(_ordered_by_foreign_key_dependency(DismodFileBase.metadata, input.split()))
    assert out == expected.split()


def test_ordering_unhappy():
    with pytest.raises(ValueError):
        next(_ordered_by_foreign_key_dependency(DismodFileBase.metadata, "nonexistent age".split()))


def test_wrong_type(base_file):
    ages = pd.DataFrame({"age": np.array(["strings", "for", "ages"])})
    base_file.age = ages
    with pytest.raises(DismodFileError):
        base_file.flush()


def test_non_existent_table(base_file):
    with pytest.raises(KeyError):
        base_file.integrands_with_an_s = pd.DataFrame()


def test_empty_table__regular_tables(base_file):
    table = base_file.empty_table("node")
    assert set(table.columns) == {"node_id", "node_name", "parent"}
    assert table.dtypes.node_id == np.int64
    assert table.dtypes.node_name == np.object
    assert table.dtypes.parent == np.int64


def test_empty_table__table_with_mutable_columns(base_file, dummy_data_row):
    table = base_file.empty_table("data")
    assert "x_sex" not in table.columns
    assert "x_s_source" not in table.columns

    base_file.data = dummy_data_row
    base_file.flush()

    table = base_file.empty_table("data")
    assert "x_sex" in table.columns
    assert "x_s_source" in table.columns


def test_attribute_access_of_empty_table(base_file):
    table = base_file.node
    assert set(table.columns) == {"node_id", "node_name", "parent"}


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

    dm_file2 = DismodFile(engine)

    dm_file2.age

    assert not dm_file2._is_dirty("age")


def test_is_dirty__not_yet_read(base_file):
    assert not base_file._is_dirty("foo_bar")


def test_add_log_has_primary_key(base_file):
    base_file.flush()
    base_file.log = pd.DataFrame(
        {
            "message_type": ["command"],
            "table_name": np.array([None], dtype=np.object),
            "row_id": np.NaN,
            "unix_time": int(round(time.time())),
            "message": ["fit_no_covariates.py"],
        }
    )
    base_file.flush()


def test_dmfile_read(base_file, engine):
    ages = base_file.age
    times = base_file.time
    base_file.flush()

    dm_file2 = DismodFile(engine)
    assert ages.sort_index("columns").equals(dm_file2.age.sort_index("columns"))
    assert times.sort_index("columns").equals(dm_file2.time.sort_index("columns"))


def test_reading_modified_columns(base_file, engine):
    base_file.flush()
    base_file.age.loc[0, "age"] *= 2
    ages = base_file.age.copy()
    base_file.flush()

    dm_file2 = DismodFile(engine)
    assert ages.sort_index("columns").equals(dm_file2.age.sort_index("columns"))


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
            "nonnullable_column": [1, 2, 3],
        }
    )
    _validate_data(DummyTable.__table__, data)


def test_validate_data__bad_integer():
    data = pd.DataFrame({"integer_column": np.array(["1", "2", "3"], dtype=np.str), "nonnullable_column": [1, 2, 3]})
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


def test_write_covariate_column__success(base_file):
    new_data = pd.DataFrame(
        {
            "data_name": "foo",
            "integrand_id": 1,
            "density_id": 1,
            "node_id": 1,
            "weight_id": 1,
            "hold_out": 0,
            "meas_value": 0.0,
            "meas_std": 0.0,
            "eta": np.nan,
            "nu": np.nan,
            "age_lower": 0,
            "age_upper": 10,
            "time_lower": 1990,
            "time_upper": 2000,
            "x_s_source": 0,
            "x_sex": 2.0,
        },
        index=[0],
    )
    base_file.data = new_data
    base_file.flush()


@pytest.fixture
def dummy_data_row():
    return pd.DataFrame(
        {
            "data_name": "foo",
            "integrand_id": 1,
            "density_id": 1,
            "node_id": 1,
            "weight_id": 1,
            "hold_out": 0,
            "meas_value": 0.0,
            "meas_std": 0.0,
            "eta": np.nan,
            "nu": np.nan,
            "age_lower": 0,
            "age_upper": 10,
            "time_lower": 1990,
            "time_upper": 2000,
            "x_s_source": 0,
            "x_sex": 2.0,
        },
        index=[0],
    )


def test_read_covariate_column__success(base_file, dummy_data_row):
    base_file.data = dummy_data_row
    base_file.flush()

    new_dm = DismodFile(base_file.engine)
    assert list(new_dm.data.x_s_source) == [0]
    assert list(new_dm.data.x_sex) == [2.0]


def test_write_covariate_column__bad_name(base_file, dummy_data_row):
    dummy_data_row = dummy_data_row.rename(columns={"x_sex": "not_x_sex"})
    base_file.data = dummy_data_row
    with pytest.raises(ValueError) as excinfo:
        base_file.flush()
    assert "not_x_sex" in str(excinfo.value)


def test_write_covariate_column__schema_changes_are_isolated(dummy_data_row):
    dm_file = DismodFile(_get_engine(None))
    dm_file.data = dummy_data_row
    dm_file.flush()

    dm_file2 = DismodFile(_get_engine(None))
    table = dm_file2.data
    assert "x_sex" not in table
