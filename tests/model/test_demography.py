import numpy as np
from cascade.stats import DemographicInterval


def test_demog_interval_basic():
    nx_list = [7 / 365, 21 / 365, (365 - 28) / 365, 4, 5, 5, 5]
    di = DemographicInterval(nx_list)
    assert len(di) == len(nx_list)
    assert di.bound.shape[0] == len(nx_list) + 1
    assert np.allclose(di.start[1:], np.cumsum(nx_list)[:-1])
    assert np.allclose(di.finish, np.cumsum(nx_list))
    assert abs(di.omega - 20) < 1e-6

    di35 = di[3:5]
    assert np.allclose(di35.nx, [4, 5])
    print(f"di nx {di35.nx} di start {di35.start} di finish {di35.finish}")
    assert np.allclose(di35.start, [1, 5])


def test_demog_interval_overlap():
    nx1_list = [7 / 365, 21 / 365, (365 - 28) / 365, 4, 5, 5, 5]
    di1 = DemographicInterval(nx1_list)
    nx2_list = [1] * 20
    di2 = DemographicInterval(nx2_list)
    print(di1[3:5], di2[:3])
    dio1 = di2.overlaps_with(di1[3:5])
    assert dio1.start[0] == 1
    assert len(dio1) == 9
    dio2 = di1.overlaps_with(di2[:3])
    assert dio2.start[0] == 0
    assert dio2.finish[-1] == 5
    assert len(dio2) == 4
