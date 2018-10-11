import cascade.core.db
from cascade.core.db import ModuleProxy, DatabaseSandboxViolation

import pytest


@pytest.fixture
def save_access():
    """Testing will use this value, so we save whatever was
    set in order not to mess with testing."""
    saved = cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS
    yield cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS
    cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = saved


def test_any_proxy(save_access):
    cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = False
    m = ModuleProxy("math")
    assert m.floor(3.2) == 3.0


def test_disable_proxy(save_access):
    cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = False
    m = ModuleProxy("pathlib")
    m.Path("/ihme/somewhere")
    cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = True
    with pytest.raises(DatabaseSandboxViolation):
        m.Path("/ihme/code")


def test_disable_all(save_access):
    cascade.core.db.BLOCK_SHARED_FUNCTION_ACCESS = True
    with pytest.raises(DatabaseSandboxViolation):
        cascade.core.db.db_queries.get_cause_metadata("the values", "don't matter")


def test_ihme_access(ihme):
    """
    This test should be skipped normally and execute (and pass)
    when the --ihme flag is used.
    """
    df = cascade.core.db.db_queries.get_age_metadata(age_group_set_id=12, gbd_round_id=5)
    assert not df.empty
