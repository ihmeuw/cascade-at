import logging
import sys
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

import networkx as nx

LOGGER = logging.getLogger(__name__)


def components_referenced_by(source_component, directory_of_package):
    """Find components used by a particular component.

    A component is one level down, like ``cascade.executor``.
    This tests what components are used by loading every single
    Python file within subdirectories of that directory and then
    looking at what was loaded. Can be destructive to imports
    if things were already loaded.
    """
    root_package = source_component.split(".")[0]
    component_directory = directory_of_package / source_component.replace(".", "/")
    for p in component_directory.glob("**/*.py"):
        rel_to_package = p.relative_to(directory_of_package)
        import_module(str(rel_to_package)[:-3].replace("/", "."))
    components_in_package = [_ for _ in sys.modules.keys()
                             if _.startswith(f"{root_package}.")]
    for purge in components_in_package:
        del sys.modules[purge]
    del sys.modules[root_package]
    return components_in_package


def component_in_package(directory_of_package, package_name):
    for init in (directory_of_package / package_name).glob("*/__init__.py"):
        yield str(init.parent.relative_to(directory_of_package)).replace("/", ".")


def module_dependencies(package_root, package_name):
    components = dict()
    for com in component_in_package(package_root, package_name):
        refs = components_referenced_by(com, package_root)
        other_refs = (ex for ex in refs if not ex.startswith("com"))

        LOGGER.debug(f"{com}:")
        components[com] = set()
        for r in sorted(other_refs):
            LOGGER.debug(f"\t{r}")
            base = ".".join(r.split(".")[:2])
            if base != com:
                components[com].add(base)
        LOGGER.debug("")
    return components


def dependencies_are_directed(package_root, package_name):
    """
    Tells you whether there are cycles in the dependency
    graph of components.

    Args:
        package_root (Path): The ``src`` directory.
        package_name (str): "cascade".

    Returns:
        bool
    """
    dependencies = module_dependencies(Path(package_root), package_name)
    component_graph = nx.DiGraph()
    for component_name, deps in dependencies.items():
        component_graph.add_edges_from([(component_name, dep) for dep in deps])
    no_cycles = nx.is_directed_acyclic_graph(component_graph)
    if not no_cycles:
        print(f"nodes {nx.nodes(component_graph)} end")
    return no_cycles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("package_root", type=Path)
    parser.add_argument("package_name")
    args = parser.parse_args()
    directed = dependencies_are_directed(args.package_root, args.package_name)
    exit(0 if directed else 3776)


def test_find_components():
    c = components_referenced_by("cascade.core", Path("src"))
    assert "cascade.model" in c
    assert "cascade.core" in c


def test_component():
    component_list = list(component_in_package(Path("src"), "cascade"))
    assert "cascade.testing_utilities" in component_list
    assert "cascade.model" in component_list
