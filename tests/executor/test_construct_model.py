from pathlib import Path
import pickle
from types import SimpleNamespace

from numpy.random import RandomState

from cascade.executor.construct_model import construct_model
from cascade.executor.create_settings import create_local_settings
from cascade.model.session import Session
from cascade.input_data.db.locations import location_hierarchy_to_dataframe


def test_construct_model_fair(dismod, tmp_path):
    lose_file = False
    filename = tmp_path / "z.db" if lose_file else "model_fair.db"
    rng = RandomState(424324)
    for i in range(10):
        rng_state = rng.get_state()
        local_settings, locations = create_local_settings(rng)
        data = SimpleNamespace()
        data.locations = locations
        model = construct_model(data, local_settings)

        assert len(model.rate.keys()) > 0
        session = Session(location_hierarchy_to_dataframe(locations),
                          parent_location=1, filename=filename)
        try:
            session.setup_model_for_fit(model)
        except AssertionError:
            pickle.dump(rng_state, Path(f"fail_state{i}.pkl").open("wb"))
            raise
