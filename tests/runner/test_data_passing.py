from copy import deepcopy

import pytest

from cascade.executor.execution_context import make_execution_context
from cascade.runner.data_passing import DbFile, PandasFile, ShelfFile


@pytest.fixture
def context(tmp_path):
    ec = make_execution_context(
        base_directory=tmp_path,
        modelable_entity_id=123,
        model_version_id=9876,
    )
    return dict(ec=ec, tmp=tmp_path)


def test_file_path_constructed(context):
    # No validation here
    db = DbFile(context["ec"], "my.db", location_id=34, sex="male")
    relative_path = db.path.relative_to(context["tmp"])
    assert relative_path.name == "my.db"
    assert not relative_path.is_absolute()
    assert isinstance(db.validate(), str)
    # Removing without a file does nothing
    db.remove()


def test_file_path_subdir_constructed(context):
    """Ensure that relative paths in subdirectories can be written."""
    db = DbFile(context["ec"], "subdir/my.db", location_id=34, sex="male")
    relative_path = db.path.relative_to(context["tmp"])
    assert relative_path.name == "my.db"
    assert not relative_path.is_absolute()
    assert isinstance(db.validate(), str)


def test_file_path_validates(context):
    required_tables = ["one", "two"]
    db = DbFile(
        context["ec"], "my.db", location_id=34, required_tables=required_tables)
    # Test for missing file.
    assert db.validate()
    db.mock()
    # Show mocking works.
    assert db.validate() is None
    # And then remove it.
    db.remove()
    assert db.validate() is not None


def test_file_path_anti_validates(context):
    ec = context["ec"]
    given = ["one", "two"]
    db = DbFile(ec, "my.db", location_id=34, sex="female", required_tables=given)
    # Mock one type of file.
    db.mock()

    # But validate a different one to see that it fails.
    required = ["one", "two", "three"]
    db = DbFile(ec, "my.db", location_id=34, sex="female", required_tables=required)
    validation = db.validate()
    assert validation is not None


def test_pandas_no_validate(context):
    ec = context["ec"]
    # No validation
    pdf = PandasFile(ec, "my.hdf", location_id=29, sex="both")
    assert pdf.validate()
    pdf.mock()
    assert not pdf.validate()


def test_pandas_validate_happy(context):
    ec = context["ec"]
    datasets = dict(
        priors=["integrand", "location", "value"],
        data=["integrand", "stdev"],
    )
    pdf = PandasFile(ec, "my.hdf", location_id=29, required_frames=datasets)
    assert pdf.validate()
    pdf.mock()
    assert not pdf.validate()


def test_pandas_validate_missing_dataset(context):
    ec = context["ec"]
    datasets = dict(
        priors=["integrand", "location", "value"],
        data=["integrand", "stdev"],
    )
    pdf = PandasFile(ec, "my.hdf", location_id=29, required_frames=datasets)
    missing_one = dict(priors=datasets["priors"])
    less = PandasFile(ec, "my.hdf", location_id=29, required_frames=missing_one)
    less.mock()
    assert pdf.validate() is not None


def test_pandas_validate_missing_columns(context):
    ec = context["ec"]
    datasets = dict(
        priors=["integrand", "location", "value"],
        data=["integrand", "stdev"],
    )
    pdf = PandasFile(ec, "my.hdf", location_id=29, required_frames=datasets)
    missing_col = deepcopy(datasets)
    missing_col["priors"].remove("value")
    less = PandasFile(ec, "my.hdf", location_id=29, required_frames=missing_col)
    less.mock()
    assert pdf.validate() is not None


def test_shelf_no_validate(context):
    ec = context["ec"]
    shelf = ShelfFile(ec, "my.shelf", location_id=17)
    assert shelf.validate() is not None
    # Create an empty file, but it's not the base filename.
    # Shelf files open a .dat, .bak, and .dir file.
    base_path = shelf.path
    (base_path.parent / (base_path.name + ".dat")).open("w").close()
    # An empty file should validate fine.
    assert shelf.validate() is None


def test_shelf_happy(context):
    ec = context["ec"]
    keys = {"hi", "there"}
    shelf = ShelfFile(ec, "my.shelf", location_id=17, required_keys=keys)
    assert shelf.validate() is not None
    shelf.mock()
    assert shelf.validate() is None


def test_shelf_remove(context):
    ec = context["ec"]
    keys = {"hi", "there"}
    shelf = ShelfFile(ec, "my.shelf", location_id=4, sex="female", required_keys=keys)
    shelf.mock()
    assert shelf.validate() is None
    shelf.remove()
    assert shelf.validate() is not None
