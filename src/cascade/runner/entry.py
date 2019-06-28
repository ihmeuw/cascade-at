import networkx as nx


def execution_ordered(graph):
    """For either a recipe graph or a task graph, this orders the nodes
    such that they go depth-first. This is chosen so that the data
    has the most locality during computation."""
    assert "root" in graph.graph, "Expect to find G.graph['root']"
    return nx.dfs_preorder_nodes(graph, graph.graph["root"])


def run_job_graph(job_graph, sub_graph, backend, continuation):
    run_graph = nx.subgraph(job_graph, sub_graph)
    assert nx.is_connected(run_graph.to_undirected())
    if backend == "single_process":
        run_single_process(run_graph, continuation)


def run_single_process(run_graph, continuation):
    assert not continuation
    for node in execution_ordered(run_graph):
        for run_idx in range(node.multiplicity):
            node(run_idx)


def run_qsub(run_graph, continuation):
    assert not continuation
    for node in execution_ordered(run_graph):
        if node.multiplicity > 1:
            pass  # task array
        else:
            pass  # single job
