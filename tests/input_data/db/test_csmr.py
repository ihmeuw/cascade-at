from cascade_at.executor.execution_context import make_execution_context
from cascade_at.inputs.csmr import _csmr_in_t3


def test_csmr_check_in_t3_by_location(ihme):
    ec = make_execution_context()
    # This should be a list of location ids.
    # TODO: Re-instate this test when we get model results and input
    #  data uploaded for CSMR.
    # locs = _csmr_in_t3(ec, 267245)
    # assert locs == [80]

    # If the model version id doesn't exist, return an empty list.
    locs = _csmr_in_t3(ec, -3)
    assert not locs
