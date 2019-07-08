from cascade.core.log import getLoggers
from .grid_process import run_check

CODELOG, MATHLOG = getLoggers(__name__)


def max_run_time_on_queue(queue_name):
    qconf_key_value = run_check("qconf", ["-sq", queue_name])
    return [x.split() for x in qconf_key_value.splitlines()
            if x.startswith('h_rt')][0][1]


def template_to_args(template):
    """This encodes the consistent rule for qsub's flag system.
    Represent arguments to qsub with a dictionary where the keys are the
    flags and the values are either None, for don't include this flag,
    or a value, or a dictionary of key=value pairs.

    Why do this? It's easier to modify than a long string and puts
    no requirements on making the code know all of the flags.

    Args:
        template: A dictionary of lists, values, and dictionaries.

    Returns:
        List[str]: Suitable for passing to qsub.
    """
    args = []
    for flag, value in template.items():
        if value is None:
            args.append(f"-{flag}")
        elif isinstance(value, bool):
            args.extend([f"-{flag}", str(value).upper()])
        elif isinstance(value, dict):
            kv_pairs = list()
            for tag, amount in value.items():
                if amount is None:
                    kv_pairs.append(str(tag))
                elif isinstance(amount, bool):
                    kv_pairs.append(f"{tag}={str(amount).upper()}")
                else:
                    kv_pairs.append(f"{tag}={amount}")
            if kv_pairs:
                args.extend([f"-{flag}", ",".join(kv_pairs)])
            else:
                pass  # nothing to add.
        else:
            args.extend([f"-{flag}", str(value)])
    return args


def qsub_template():
    """
    Basic template for qsub. This means that any flags that can
    have multiple copies are already included in the data structure.
    So you can do ``template["l"]["intel"]`` without having
    to check that "l" exists.

    .. code::

        template = qsub_template()
        template["q"] = "all.q"
        template["P"] = "proj_forecasting"
        template["l"]["h_rt"] = "12:00:00"
        args = template_to_args()
        assert "-q all.q" in " ".join(args)

    """
    # These are all the flags that can be repeated.
    return {flag: dict() for flag in ["l", "F", "pe", "U", "u"]}


def qsub(template, command):
    """
    Runs a qsub command with a template.

    We can either try to put a super-thoughtful interface on qsub, or we
    let the user manage its arguments. This focuses on making it a little
    easier to manage arguments with the template.

    Args:
        template: Suitable for `template_to_args`.
        command (List[str]): A list of strings to pass to qsub.

    Returns:
        str: The model version ID. It's a str because it isn't an int.
        Can you add 37 to it? No. Is it ordered? That's not guaranteed.
        Does it sometimes have a ".1" at the end? Yes.
        That makes it a string.
    """
    str_command = [str(x) for x in command]
    formatted_args = template_to_args(template)
    args = ["-terse"] + formatted_args + str_command
    return run_check("qsub", args)
