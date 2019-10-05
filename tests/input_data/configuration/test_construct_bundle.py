from collections import defaultdict

import pandas as pd
import pytest
from numpy import nan, isnan

from cascade.input_data.configuration.construct_bundle import (
    bundle_to_observations, normalized_bundle_from_database
)
from cascade.executor.execution_context import make_execution_context

@pytest.mark.skip
def test_bundle_to_observations__global_eta():
    df = pd.DataFrame(
        {
            "location_id": 90,
            "measure": "Tincidence",
            "age_lower": 0,
            "age_upper": 120,
            "time_lower": 1980,
            "time_upper": 2018,
            "mean": 0.1,
            "lower": 0.05,
            "upper": 0.15,
            "sex_id": 3,
            "seq": 0,
            "hold_out": 0,
        },
        index=[0],
    )

    eta = dict(Tincidence=nan)
    density = dict(Tincidence="gaussian")
    nu = defaultdict(lambda: 5.0)
    observations = bundle_to_observations(df, 90, eta, density, nu)
    assert isnan(observations.eta[0])

    eta = dict(Tincidence=1e-2)
    density = dict(Tincidence="gaussian")
    observations = bundle_to_observations(df, 90, eta, density, nu)
    assert observations.eta[0] == 1e-2


def test_bundle_from_database(ihme):
    import pdb
    pdb.set_trace()
    ec = make_execution_context()
    bundle = normalized_bundle_from_database(ec, 264749)
    assert bundle is not None
    parent_location_id = 90
    eta = defaultdict(lambda: 5e-3)
    eta["Tincidence"] = 1e-2
    eta["mtother"] = 7e-3
    eta["prevalence"] = 0.1
    density = defaultdict(lambda: "gaussian")
    density["Tincidence"] = "laplace"
    density["mtexcess"] = "log_students"
    nu = defaultdict(lambda: 5.0)
    observations = bundle_to_observations(bundle, parent_location_id, eta, density, nu)
    assert len(observations) == len(bundle)

    etas = observations.eta.unique()
    assert len(etas) == 3
    for eval in [5e-3, 1e-2, 0.1]:
        assert (etas == eval).any()

    densities = observations.density.unique()
    assert len(densities) == 2
    for dname in ["gaussian", "laplace"]:
        assert (densities == dname).any()
