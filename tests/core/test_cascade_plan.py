from cascade.core.parameters import _ParameterHierarchy
from cascade.core.cascade_plan import CascadePlan
from cascade.testing_utilities import make_execution_context


def test_create(ihme):
    ec = make_execution_context(location_id=0, gbd_round_id=5)
    settings = _ParameterHierarchy(
        model={"split_sex": 3, "drill_location": 6},
        policies=dict(),
    )
    c = CascadePlan.from_epiviz_configuration(ec, settings)
    assert len(c.task_graph.nodes) == 2
    print(c.task_graph.nodes)
