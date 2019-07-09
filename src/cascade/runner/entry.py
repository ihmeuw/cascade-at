import networkx as nx

from cascade.core.log import getLoggers
from cascade.executor.execution_context import application_config
from .submit import max_run_time_on_queue, qsub

CODELOG, MATHLOG = getLoggers(__name__)


def execution_ordered(graph):
    """For either a recipe graph or a task graph, this orders the nodes
    such that they go depth-first. This is chosen so that the data
    has the most locality during computation. It's not strictly
    depth-first, but depth-first, given that all predecessors must
    be complete before a node executes."""
    assert "root" in graph.graph, "Expect to find G.graph['root']"
    possible = [graph.graph["root"]]
    seen = set()
    in_order = list()
    while possible:
        node = possible.pop()
        parents_must_complete = set(graph.predecessors(node))
        if node not in seen and not parents_must_complete - seen:
            seen.add(node)
            in_order.append(node)
            for successor in graph.successors(node):
                possible.append(successor)

    return in_order


def run_job_graph(work, backend, continuation):
    run_graph = nx.subgraph(work["job_graph"], work["sub_graph"])
    assert nx.is_connected(run_graph.to_undirected())
    if backend == "single_process":
        run_single_process(work, run_graph, continuation)
    elif backend == "grid_engine":
        run_qsub(run_graph, continuation)
    else:
        raise RuntimeError(f"Cannot identify backend {backend}.")


def run_single_process(work, run_graph, continuation):
    assert not continuation
    local_cache = dict()
    for node in execution_ordered(run_graph):
        job = run_graph.nodes[node]["job"]
        for run_idx in range(job.multiplicity):
            job(
                work["execution_context"],
                run_graph.nodes[node]["local_settings"],
                local_cache,
            )


def run_mock(work, run_graph, continuation):
    assert not continuation
    for node in execution_ordered(run_graph):
        job = run_graph.nodes[node]["job"]
        for run_idx in range(job.multiplicity):
            job.mock_run(work["execution_context"])


def run_qsub(work, run_graph, continuation, mvid=None):
    assert not continuation
    mvid = mvid if mvid else "7714565980"
    grid_engine_job = dict()
    parameters = application_config()["GridEngine"]
    main_queue = parameters["queues"][0]
    max_runtime = max_run_time_on_queue(main_queue)
    for node in execution_ordered(run_graph):
        job = run_graph.nodes[node]["job"]
        memory = f"{job.memory_resource}G"
        threads = f"{job.thread_resource}"
        template = dict(
            N=f"dmat_{mvid}_{node}",
            q=main_queue,
            l=dict(h_rt=max_runtime, m_mem_free=memory, fthread=threads),
            P=parameters["project"],
            j="y",
            b="y",
        )
        holds = [grid_engine_job[parent]
                 for parent in run_graph.predecessors(node)]
        if holds:
            template["h"] = holds
        if node.multiplicity > 1:
            template["t"] = f"1-{node.multiplicity}"  # task array
        command = ["/bin/bash", "--noprofile", "--norc", rooted_script, mvid, epi_environment]
        job_id = qsub(template, command)
        grid_engine_job[node] = job_id
