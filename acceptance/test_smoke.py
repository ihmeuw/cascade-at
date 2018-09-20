"""
Ensure that applications are installed and runnable.
"""
from subprocess import run


def test_dismod_smoke():
    """
    Tests whether Dismod can be found from the dmdismod script.
    """
    run("dmdismod --help".split(), check=True)


def test_csv2db_smoke():
    """
    Tests whether dmcsv2db is installed into the current environment.
    """
    run("dmcsv2db --help".split(), check=True)
