"""
Functions that interpret the return code, log, stdout and stderr
from Dismod-AT. These are rules that may change as Dismod-AT
output is edited.

 * Dismod-AT writes "end <command>" to the log just before exit(0).
   All other error messages are on stderr.

 * Many errors use error_exit, so that they write to the db log,
   print to stderr, and then exit(1).

 * The only non-zero exit code chosen by Dismod-AT is exit(1),
   which means other return codes are from signals.

 * Messages may come from cppad_mixed or from Ipopt, which can
   both print to stderr. The Ipopt messages are in
   ``./Ipopt/src/Interfaces/IpIpoptApplication.cpp``.
   Ipopt does have return codes, but they seem to be buried in CppAD.

"""
import re
from cascade.dismod import DismodATException
from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)

RE_EXIT = re.compile(r"EXIT: (.*)")
RE_ITERATIONS = re.compile(r"Number of Iterations\W+ (\d+)")


IPOPT_PERFECT = {
    "Optimal Solution Found",
    "Solved To Acceptable Level",
    "Feasible point for square problem found",
    "Stopping optimization at current point as requested by user",
}
IPOPT_NO_RESULTS_TO_SEE = {
    "Not enough memory",
    "Invalid option encountered",
    "Some uncaught Ipopt exception encountered",
    "Restoration Failed!",
    "INTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors",
    "Problem has too few degrees of freedom",
    "Invalid option encountered",
    "Problem has only fixed variables and constraints are infeasible",
    "Problem has inconsistent variable bounds or constraint sides",
    "Some uncaught Ipopt exception encountered",
}
IPOPT_EXAMINE_RESULTS = {
    "Maximum Number of Iterations Exceeded",
    "Maximum CPU time exceeded",
    "Search Direction is becoming Too Small",
    "Iterates diverging; problem might be unbounded",
    "Error in step computation (regularization becomes too large?)!",
    "Converged to a point of local infeasibility. Problem may be infeasible",
    "Invalid number in NLP function or derivative detected",
}


def get_fit_output(stdout):
    """
    After running fit, use this to get the results from
    stdout.

    Args:
        stdout (str): Stdout as a string, not bytes.

    Returns:
        (str, str, int): The first member is one of the strings
        "perfect", "failed", "examine", or "unknown",
        and the second string is the EXIT message from Ipopt.
        The "examine" means that Ipopt likely produced a result
        that will be helpful for improving the next run, so we should
        continue.
    """
    ipopt_class, ipopt_exit = _fit_ipopt_out(stdout)
    iter_match = RE_ITERATIONS.search(stdout)
    iteration_cnt = 0
    if iter_match:
        try:
            iteration_cnt = int(iter_match.group(1))
        except ValueError:
            CODELOG.warning(f"Failed to translate {iter_match} to int")
    return ipopt_class, ipopt_exit, iteration_cnt


def check_command(command, log, return_code, stdout, stderr):
    """
    This raises an exception if something went wrong. Otherwise
    it returns nothing. The rules are:

    1. An out-of-memory condition is a MemoryError.
    2. Any nonzero exit is an error, *except iterations exceeded.*
       That indicates that Ipopt reached its limit, but that's OK.

    Args:
        command (str|List[str]): The command that was last run.
        log (pd.DataFrame): The file's log messages in a dataframe.
        return_code (int): Unix process return code.
        stdout (str): Stdout as a string, not bytes.
        stderr (str): Stderr as a string, not bytes.

    """
    _failure_for_any_dismod_command(return_code, stderr, stdout)
    if isinstance(command, str):
        base_command = command
    else:
        base_command = command[0]
    if base_command == "fit":
        ipopt_class, ipopt_exit = _fit_ipopt_out(stdout)
        if ipopt_class != "examine":
            _claimed_to_be_complete(log, "fit", stderr)
        else:
            CODELOG.info(f'Fit ended with "{ipopt_exit}" so not failing.')
    else:
        _claimed_to_be_complete(log, base_command, stderr)


def _failure_for_any_dismod_command(return_code, stderr, stdout):
    out_of_memory_sentinel = "std:bad_alloc"
    if out_of_memory_sentinel in stdout or out_of_memory_sentinel in stderr:
        raise MemoryError("Dismod-AT ran out of memory")
    dismods_only_return_values = {0, 1}
    if return_code not in dismods_only_return_values:
        raise DismodATException(
            f"Dismod-AT exited with return code {return_code}.\n{stderr}")


def _claimed_to_be_complete(log, command, stderr):
    if len(log) == 0 or f"end {command}" not in log.message.iloc[-1]:
        raise DismodATException(
            f"Dismod-AT failed to complete '{command}' command\n{stderr}")


def _fit_ipopt_out(stdout):
    ipopt_exit_match = RE_EXIT.search(stdout)
    if ipopt_exit_match:
        ipopt_exit = ipopt_exit_match.groups(1)[0]
        if any(p in ipopt_exit for p in IPOPT_PERFECT):
            ipopt_class = "perfect"
        elif any(p in ipopt_exit for p in IPOPT_NO_RESULTS_TO_SEE):
            ipopt_class = "failed"
        elif any(p in ipopt_exit for p in IPOPT_EXAMINE_RESULTS):
            ipopt_class = "examine"
        else:
            ipopt_class = "unknown"
    else:
        ipopt_class = "unknown"
        ipopt_exit = "could not find Ipopt exit status"
    return ipopt_class, ipopt_exit
