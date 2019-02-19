from types import SimpleNamespace

from numpy.random import RandomState

from cascade.executor.construct_model import construct_model
from cascade.executor.create_settings import create_local_settings


def test_construct_model_fair():
    rng = RandomState(424324)
    for i in range(10):
        local_settings, locations = create_local_settings(rng)
        data = SimpleNamespace()
        data.locations = locations
        model = construct_model(data, local_settings)

        assert len(model.rate.keys()) > 0
