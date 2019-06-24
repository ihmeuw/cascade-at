"""
The graph of tasks is built in layers.

1. GBD shared functions give the location hierarchy for ALL locations.
2. CascadePlan converts those into a graph of estimations
   that have location-local settings.
3. Then they become separate sub-graphs for each location.
4. Those separate sub-graphs are assigned stages.
5. The graph-of-subgraphs becomes a single graph of executable stages.

.. code:

    transform_graph = transform_graph_from_settings(locations, settings, args)
    cascade_plan = CascadePlan(locations, settings, args)

"""


def connect_subtasks(transform_graph):
    task_graph = None
    return task_graph
