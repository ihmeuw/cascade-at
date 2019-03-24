"""
Plots dependencies among components.
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from subprocess import run

from graphviz import Digraph

from cascade.testing_utilities.component_dependencies import module_dependencies

LOGGER = logging.getLogger("pystructure.graphviz_plot")


def dot_graph(deps):
    dot = Digraph(comment="Component Graph")
    for source_package in deps.keys():
        dot.node(source_package)
    for sinks in deps.values():
        for sink_package in sorted(sinks):
            dot.node(sink_package)
    for node, targets in deps.items():
        for t in targets:
            dot.edge(node, t)
    return dot


def plot(deps):
    dot = dot_graph(deps)
    dot.render("components")
    run("convert -density 300 -depth 8 -background white components.pdf -resize 30% -trim components.png".split())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--package-name")
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()

    all_dependencies = module_dependencies(args.package_root, args.package_name)
    plot(all_dependencies)
