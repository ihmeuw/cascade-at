from pathlib import Path
from subprocess import run

from pkg_resources import get_distribution


def test_dependencies_directed():
    """Checks whether there are cycles in the graph of dependencies

    This codebase defines a component as one directory down, so
    ``cascade.core`` is at the bottom and ``cascade.executor`` at
    the top of the dependency graph. If this test fails, then there
    is a cycle in that graph of dependencies among the subdirectories one
    level down. That can lead to very difficult-to-debug import problems
    later. Fix it now.
    """
    package_root = Path(get_distribution("cascade").location)
    # Run this as a subprocess because it loads and deletes modules, which
    # sincerely messes with pytest's ability to construct mock objects
    # in conftest.py.
    result = run(f"python -m cascade.testing_utilities.component_dependencies {package_root} cascade".split())
    # That command won't run at all if the library isn't unpacked into
    # source files, so ignore when it fails entirely.
    if result.returncode in {0, 3776}:
        assert result.returncode == 0, f"{result.stdout} {result.stderr}"


def test_is_namespace_package():
    """
    Cascade is a namespace package. The ``cascade_config`` package is in
    the same namespace. Namespace packages don't have an ``__init__.py``
    in the base directory. That's it. That's all you have to do is not
    put an ``__init__.py`` in the base directory, and you're good.
    """
    package_root = Path(get_distribution("cascade").location)
    print(package_root)
    assert not (package_root / "cascade" / "__init__.py").exists()
