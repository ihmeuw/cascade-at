from types import SimpleNamespace

import pytest

from cascade.executor.cascade_plan import EstimationParameters, IHMEDataParameters
from cascade.executor.estimate_location import retrieve_data
from cascade.testing_utilities import make_execution_context


@pytest.fixture
def local_settings():
    ihme_params = IHMEDataParameters(5, 23514, 267770)
    settings = SimpleNamespace()
    settings.model = SimpleNamespace()
    settings.model.bundle_id = 23472
    settings.policies = SimpleNamespace()
    settings.policies.age_group_set_id = 12
    settings.policies.with_hiv = True
    policies = None
    return EstimationParameters(settings, policies, ihme_params, [101], 101, None)


@pytest.mark.skip()
def test_retrieve_data(ihme):
    ec = make_execution_context()

    retrieve_data(ec, local_settings)
