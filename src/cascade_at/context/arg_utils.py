from typing import Dict, Any, List, Union

from cascade_at.core import CascadeATError


type_mappings = {
    'int': int,
    'str': str,
    'float': float
}


def list2string(x: List[Union[int, str, float]]) -> str:
    return " ".join([str(x) for x in x])


class OptionParsingError(CascadeATError):
    """Raised when there is an issue parsing an option dict for the command line."""
    pass


def parse_options(option_list: List[str]) -> Dict[str, Union[int, float, str]]:
    """
    Parse a key=value=type command line arg
    that comes in a list.

    Returns
    -------
    Dictionary of options with the correct types.
    """
    d = dict()
    for o in option_list:
        o = o.split('=')
        if len(o) != 3:
            raise OptionParsingError("Not enough elements in the parsed options. Need 3 elements.")
        key = o[0]
        val = o[1]
        if o[2] not in type_mappings:
            raise OptionParsingError(f"Unknown option type {o[2]}.")
        type_func = type_mappings[o[2]]
        d.update({key: type_func(val)})
    return d


def encode_options(options: Dict[str, Any]) -> List[str]:
    """
    Encode an option dict into a command line string that cascade_at can understand.

    Returns
    -------
    List of strings that can be passed to the command line..
    """
    d = list()
    rev_dict = {v: k for k, v in type_mappings.items()}
    for k, v in options.items():
        t = type(v)
        if t not in rev_dict:
            raise OptionParsingError(f"Unknown option type {t}.")
        arg = f'{k}={v}={rev_dict[t]}'
        d.append(arg)
    return d


def parse_commands(command_list: List[str]) -> List[str]:
    """
    Parse the dismod commands that come from command line arguments
    in a list.

    Returns
    -------
    list of commands that dismod can understand
    """
    return [' '.join(x.split('-')) for x in command_list]


def encode_commands(command_list) -> List:
    """
    Encode the commands to a DisMod database so they can be
    passed to the command line.

    Parameters
    ----------
    command_list

    Returns
    -------

    """
    return ['-'.join(x.split(' ')) for x in command_list]
