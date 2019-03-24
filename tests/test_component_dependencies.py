from pathlib import Path
from subprocess import run

from pkg_resources import get_distribution


def test_dependencies_directed():
    """Checks whether there are cycles in the graph of dependencies"""
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
    package_root = Path(get_distribution("cascade").location)
    print(package_root)
    assert not (package_root / "cascade" / "__init__.py").exists()
