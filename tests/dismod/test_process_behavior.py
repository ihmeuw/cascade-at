import pandas as pd
import pytest
from cascade.dismod.process_behavior import check_command, get_fit_output
from cascade.dismod import DismodATException


def test_successful_completion():
    log = pd.DataFrame(dict(message=["end fit fixed"]))
    check_command(["fit", "fixed"], log, 0, "EXIT: Optimal Solution Found.", "")


def test_single_string():
    log = pd.DataFrame(dict(message=["end predict"]))
    check_command("predict", log, 0, "Gibberish", "")


@pytest.mark.parametrize("cmd,log,ret,stdout", [
    ("predict", "end predict", 0, ""),  # everything perfect
    ("predict", "end predict", 1, ""),  # 1 return code OK if end predict there.
    ("fit", "end fit fixed", 0, ""),  # everything perfect.
    ("fit", "end fit fixed", 0, "\nEXIT: Feasible point for square problem found."),
    # Even a failing message from ipopt ignored if Dismod-AT log happy.
    ("fit", "end fit fixed", 0, "EXIT: Invalid option encountered."),
    # Ipopt failed in a forgivable way, and Dismod-AT said it failed, but we
    # ignore it in these cases.
    ("fit", "anything else", 1, "\nEXIT: Maximum CPU time exceeded.\n"),
])
def test_success_general(cmd, log, ret, stdout):
    log_df = pd.DataFrame(dict(message=[log]))
    check_command(cmd, log_df, ret, stdout, "")


@pytest.mark.parametrize("cmd,log,ret,stdout", [
    # wrong end message
    ("predict", "end fit fixed", 0, ""),
    # wrong end message and not a forgivable failure.
    ("fit", "end wrong end", 0, "\nEXIT: Restoration Failed!"),
    # wrong end message and not a forgivable failure.
    ("fit", "anything else", 1, "\nEXIT: Some uncaught Ipopt exception encountered"),
])
def test_process_failure_general(cmd, log, ret, stdout):
    log_df = pd.DataFrame(dict(message=[log]))
    with pytest.raises(DismodATException):
        check_command(cmd, log_df, ret, stdout, "")


STDOUT = """
 81 -2.4294118e-02 0.00e+00 7.51e-09 -11.0 5.45e-09    -  1.00e+00 5.00e-01f  2

Number of Iterations....: 81

                                   (scaled)                 (unscaled)
Objective...............:  -2.4294117959982979e-02   -2.4294117959982979e-02
otal CPU secs in IPOPT (w/o function evaluations)   =      0.192
Total CPU secs in NLP function evaluations           =      0.056

EXIT: Optimal Solution Found.

"""


def test_pulls_iterations():
    kind, message, iterations = get_fit_output(STDOUT)
    assert kind == "perfect"
    assert message == "Optimal Solution Found."
    assert iterations == 81
