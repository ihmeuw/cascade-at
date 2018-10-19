import pytest

from cascade.core import getLoggers


@pytest.mark.parametrize("module,code,math", [
    ("cascade", "cascade", "cascade.math"),
    ("cascade.core", "cascade.core", "cascade.math.core"),
    ("cascade.core.log", "cascade.core.log", "cascade.math.core.log"),
    ("__main__", "__main__", "__main__.math"),
    ("", "root", "root.math"),
])
def test_handle_bare(module, code, math):
    code_log, math_log = getLoggers(module)
    assert code_log.name == code
    assert math_log.name == math
