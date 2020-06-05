import pytest

from cascade_at.core import getLoggers


@pytest.mark.parametrize("module,code,math", [
    ("cascade_at", "cascade_at", "cascade_at.math"),
    ("cascade_at.core", "cascade_at.core", "cascade_at.math.core"),
    ("cascade_at.core.log", "cascade_at.core.log", "cascade_at.math.core.log"),
    ("__main__", "__main__", "__main__.math"),
    ("", "root", "root.math"),
])
def test_handle_bare(module, code, math):
    code_log, math_log = getLoggers(module)
    assert code_log.name == code
    assert math_log.name == math
