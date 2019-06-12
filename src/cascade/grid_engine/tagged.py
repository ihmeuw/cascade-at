from functools import wraps
from inspect import signature
from collections import defaultdict
from typing import Tuple
import networkx as nx


class GridContext:
    """Singleton for execution of grid functions."""
    live = True
    _instance = None

    def __init__(self):
        self._called = dict()
        self._previous = defaultdict(int)

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def add_call(self, job_action, args, kwargs, to_return):
        invocation = self._previous[job_action.__name__]
        self._previous[job_action.__name__] += 1
        self._called[(job_action, invocation)] = (
            args, kwargs, to_return
        )


def tag_job(job_action):
    sig = signature(job_action)
    retval = sig.return_annotation
    if retval != sig.empty:
        return_cnt = len(retval.__args__)
    else:
        return_cnt = 1

    @wraps(job_action)
    def wrapped(*args, **kwargs):
        if GridContext.live:
            return job_action(*args, **kwargs)
        else:
            to_return = tuple(object() for _r in range(return_cnt))
            GridContext().add_call(job_action, args, kwargs, to_return)
            return to_return

    return wrapped


@tag_job
def first_fit(a, b) -> Tuple[int, int]:
    return a + b, a - b


@tag_job
def simulation(a):
    return a + 7


def gather(a):
    return None


# Two issues. Separate them.
# 1. Order for calls.
# 2. What gets passed.

def single_location(location):
    dependency = nx.DiGraph()
    parent = (first_fit, location)
    dependency.add_node(parent)
    for i in range(5):
        child = (simulation, location, i)
        dependency.add_edge(parent, child, role="fit")
        dependency.add_edge(child, (gather, location), role=f"draw_source_{i}")
    return dependency


def aggregate(a):
    dependency = nx.DiGraph()
    dependency.add_node((aggregate, 0))
    return dependency


def whole_set():
    locations = nx.balanced_tree(3, 3, nx.DiGraph)
    depend_subgraphs = nx.DiGraph()
    for parent, child in nx.dfs_edges(locations, source=0):
        if parent not in depend_subgraphs:
            depend_subgraphs.add_node(parent, subgraph=single_location(parent))
        depend_subgraphs.add_node(child, subgraph=single_location(child))
        depend_subgraphs.add_edge(parent, child, role="parent")
    leaves = [
        leaf for leaf in locations
        if not nx.descendants(locations, leaf)
    ]
    for final in leaves:
        depend_subgraphs.add_node("aggregate", subgraph=aggregate(leaves))
        depend_subgraphs.add_edge(final, "aggregate")
    return depend_subgraphs


if __name__ == "__main__":
    print(whole_set().nodes())
