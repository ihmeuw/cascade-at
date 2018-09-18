import cascade.input_data.db as db
from cascade.input_data.db.module_proxy import ModuleProxy
from cascade.input_data.db import module_proxy

import pytest


@pytest.fixture
def save_access():
    """Testing will use this value, so we save whatever was
    set in order not to mess with testing."""
    saved = module_proxy.BLOCK_SHARED_FUNCTION_ACCESS
    yield module_proxy.BLOCK_SHARED_FUNCTION_ACCESS
    module_proxy.BLOCK_SHARED_FUNCTION_ACCESS = saved


def test_any_proxy(save_access):
    module_proxy.BLOCK_SHARED_FUNCTION_ACCESS = False
    m = ModuleProxy("math")
    assert m.floor(3.2) == 3.0


def test_disable_proxy(save_access):
    module_proxy.BLOCK_SHARED_FUNCTION_ACCESS = False
    m = ModuleProxy("pathlib")
    m.Path("/ihme/somewhere")
    module_proxy.BLOCK_SHARED_FUNCTION_ACCESS = True
    with pytest.raises(ModuleNotFoundError):
        m.Path("/ihme/code")


def test_disable_all(save_access):
    db.disable_databases()
    with pytest.raises(ModuleNotFoundError):
        db.db_queries.get_cause_metadata("the values", "don't matter")
