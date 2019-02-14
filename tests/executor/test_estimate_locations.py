import pytest

from cascade.executor.estimate_location import retrieve_data
from cascade.testing_utilities import make_execution_context


@pytest.mark.skip()
def test_retrieve_data(ihme):
    ec = make_execution_context()
    local_settings = None
    retrieve_data(ec, local_settings)
