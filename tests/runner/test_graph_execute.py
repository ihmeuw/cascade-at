from cascade.runner.graph_execute import execution_ordered
import networkx as nx


def test_execution_ordered():
    """Ensure that this will not execute a child until parents are done."""
    jobs = nx.DiGraph()
    jobs.add_edges_from([
        (0, 1), (0, 2), (1, 3), (2, 3)
    ])
    jobs.graph["root"] = 0
    ordered = execution_ordered(jobs)
    assert ordered.index(2) < ordered.index(3)
    possibles = [[0, 1, 2, 3], [0, 2, 1, 3]]
    assert ordered == possibles[0] or ordered == possibles[1]


def test_execution_path():
    jobs = nx.DiGraph()
    jobs.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4)
    ])
    jobs.graph["root"] = 0
    ordered = execution_ordered(jobs)
    assert ordered == [0, 1, 2, 3, 4]
