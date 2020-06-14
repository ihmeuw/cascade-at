from typing import Dict, List, Union
from argparse import ArgumentParser, Namespace

from cascade_at.core.log import get_loggers
from cascade_at.core import CascadeATError
from cascade_at.executor.args.args import _Argument

LOG = get_loggers(__name__)


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


def encode_options(options: Dict[str, Union[str, float, int]]) -> List[str]:
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


def encode_commands(command_list: List[str]) -> List[str]:
    """
    Encode the commands to a DisMod database so they can be
    passed to the command line.
    """
    return ['-'.join(x.split(' ')) for x in command_list]


def _arg_list_to_parser(arg_list: List[_Argument]) -> ArgumentParser:
    """
    Converts a list of arguments to an ArgumentParser.
    """
    parser = ArgumentParser()
    for arg in arg_list:
        parser.add_argument(arg._arg, **arg._kwargs)
    return parser


class ArgumentList:
    def __init__(self, arg_list: List[_Argument]):
        self.arg_list = arg_list

    def parse_args(self, args) -> Namespace:
        """
        Parses arguments from a list of arguments into an argument
        namespace using ArgumentParser.parse_args(). Also
        decodes potential dismod commands and options.
        """
        parser = _arg_list_to_parser(self.arg_list)
        args = parser.parse_args(args)
        if hasattr(args, 'dm_commands'):
            if args.dm_commands is not None:
                args.dm_commands = parse_commands(args.dm_commands)
            else:
                args.dm_commands = list()
        if hasattr(args, 'dm_options'):
            if args.dm_options is not None:
                args.dm_options = parse_options(args.dm_options)
            else:
                args.dm_options = dict()
        LOG.debug(f"Arguments: {args}.")
        return args

    @property
    def argument_dict(self):
        d = dict()
        for arg in self.arg_list:
            d.update(arg._to_dict())
        return d
