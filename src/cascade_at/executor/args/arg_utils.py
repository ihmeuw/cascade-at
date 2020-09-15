from typing import Dict, List, Union, Optional

from argparse import ArgumentParser, Namespace

from cascade_at.core import CascadeATError
from cascade_at.core.log import get_loggers
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


def _arg_to_flag(name: str) -> str:
    """
    Converts an argument to a flag, like model_version_id
    to --model-version-id.

    Parameters
    ----------
    name
        Argument name to convert

    Returns
    -------
    A flag string
    """
    arg = '-'.join(name.split('_'))
    return f'--{arg}'


def _flag_to_arg(flag: str) -> str:
    """
    Splits a flag that looks like --model-version-id into an
    argument that looks like model_version_id.

    Parameters
    ----------
    flag
        The flag string to split

    Returns
    -------
    An argument name
    """
    arg = flag.split('--')[1].split('-')
    arg = '_'.join(arg)
    return arg


def _arg_to_empty(name: str) -> str:
    """
    Convert an argument name to an "empty" placeholder
    for an argument to later be filled in. Used by the jobmon TaskTemplate.
    E.g. takes something that looks like "model_version_id" and
    converts it to "{model_version_id}"

    Parameters
    ----------
    name
        Argument name to convert

    Returns
    -------
    Converted name to placeholder
    """
    arg = "{" + name + "}"
    return arg


def _arg_to_command(k: str, v: Optional[Union[str, int, float]] = None):
    """
    Takes a key (k) and a value (v) and turns it into a command-line
    argument like k=model_version v=1 and returns --model-version 1.

    If empty, returns an a template command rather than the command itself
    """
    command = _arg_to_flag(k)
    if v is not None:
        command += f' {v}'
    return command


def _args_to_command(**kwargs):
    commands = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if type(v) == bool:
            if v:
                command = _arg_to_command(k=k)
            else:
                continue
        elif type(v) == list:
            command = _arg_to_command(k=k, v=list2string(v))
        else:
            command = _arg_to_command(k=k, v=v)
        commands.append(command)
    return ' '.join(commands)


class ArgumentList:
    def __init__(self, arg_list: List[_Argument]):
        """
        A class that does operations on a list of arguments.

        Parameters
        ----------
        arg_list
        """
        self.arg_list: List[_Argument] = arg_list

    def _to_parser(self) -> ArgumentParser:
        """
        Converts list of arguments to an ArgumentParser.
        """
        parser = ArgumentParser()
        for arg in self.arg_list:
            parser.add_argument(arg._flag, **arg._parser_kwargs)
        return parser

    def parse_args(self, args: List[str]) -> Namespace:
        """
        Parses arguments from a list of arguments into an argument
        namespace using ArgumentParser.parse_args(). Also
        decodes potential dismod commands and options.
        """
        parser = self._to_parser()
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

    @property
    def task_args(self) -> List[str]:
        return [_flag_to_arg(x._flag) for x in self.arg_list if x._task_arg]

    @property
    def node_args(self) -> List[str]:
        return [_flag_to_arg(x._flag) for x in self.arg_list if not x._task_arg]

    @property
    def template(self) -> str:
        """
        Creates a template of arguments from an argument list.
        Will return something that looks like
        "{argument1} {argument2}"
        """
        arguments = []
        for arg in self.arg_list:
            flag = arg._flag
            arg = _flag_to_arg(flag)
            placeholder = _arg_to_empty(arg)
            arguments.append(placeholder)
        return ' '.join(arguments)
