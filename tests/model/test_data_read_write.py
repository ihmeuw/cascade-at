from types import SimpleNamespace

import numpy as np
import pandas as pd

from cascade.model.data_read_write import (
    read_simulation_data
)


def test_read_simulation_data():
    # We need enough of the original data to assure ourselves
    # that untouched values remain untouched.
    data = pd.DataFrame(dict(
        integrand="Sincidence",
        mean=np.linspace(0.00, 0.04, 5),
        std=np.linspace(1.0, 1.4, 5),
        nu=np.linspace(0, 1, 5),
        name=["7", "9", "14", "22", "48"],
    ))
    assert len(data) == 5

    db = SimpleNamespace()
    db.data = pd.DataFrame(dict(
        data_id=[0, 1, 2, 3, 4],
        data_name=["7", "9", "14", "22", "48"],
    ))
    subset_idx = [1, 2, 4]
    db.data_subset = pd.DataFrame(dict(
        data_subset_id=range(3),
        data_id=subset_idx,
    ))
    db.data_sim = pd.DataFrame(dict(
        data_sim_id=range(6),
        simulate_index=np.repeat([0, 1], len(subset_idx)),
        data_subset_id=np.tile(range(len(subset_idx)), 2),
        data_sim_value=[21, 22, 24, 31, 32, 34],
        data_sim_delta=[2.1, 2.2, 2.4, 3.1, 3.2, 3.4],
    ))
    print(db.data_sim)

    modified_data = read_simulation_data(db, data, 0)
    assert len(modified_data) == len(data)
    assert not modified_data.iloc[subset_idx].equals(data.iloc[subset_idx])
    same_idx = list(x for x in range(5) if x not in subset_idx)
    assert modified_data.iloc[same_idx].equals(data.iloc[same_idx])
    assert {"integrand", "mean", "std", "nu", "name"} == set(c for c in modified_data.columns)
    print(data)
    print(modified_data)
    for i in subset_idx:
        assert np.isclose(modified_data.iloc[i]["mean"], 20 + i)
        assert np.isclose(modified_data.iloc[i]["std"], 2 + i / 10)
        assert np.isclose(modified_data.iloc[i]["nu"], i / 4)
