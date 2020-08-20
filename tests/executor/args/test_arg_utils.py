import numpy as np
import pytest

from cascade_at.executor.args.arg_utils import parse_options, parse_commands
from cascade_at.executor.args.arg_utils import encode_options, encode_commands, OptionParsingError
from cascade_at.executor.args.arg_utils import ArgumentList, list2string
from cascade_at.executor.args.args import IntArg, BoolArg, ListArg, ModelVersionID
from cascade_at.executor.args.args import DmOptions, DmCommands
from cascade_at.executor.args.arg_utils import _arg_to_flag, _args_to_command
from cascade_at.executor.args.arg_utils import _flag_to_arg, _arg_to_empty, _arg_to_command


def test_list2string():
    assert list2string(['foo', 'bar']) == 'foo bar'


def test_arg_to_flag():
    assert _arg_to_flag("model_version_id") == '--model-version-id'


def test_flag_to_arg():
    assert _flag_to_arg("--model-version-id") == "model_version_id"


def test_arg_to_empty():
    assert _arg_to_empty("model_version_id") == "{model_version_id}"


def test_argument_list():
    al = ArgumentList([IntArg('--foo'), BoolArg('--bar')])
    args = al.parse_args('--foo 1 --bar'.split())
    assert args.foo == 1
    assert args.bar


def test_argument_list_template():
    al = ArgumentList([IntArg('--foo-bar'), BoolArg('--bar')])
    assert al.template == '{foo_bar} {bar}'


def test_argument_list_task_args():
    arg1 = IntArg('--foo')
    al = ArgumentList([arg1, ModelVersionID()])
    assert al.task_args == ['model_version_id']
    assert al.node_args == ['foo']


def test_argument_list_lists():
    al = ArgumentList([ListArg('--foo', type=int)])
    args = al.parse_args('--foo 1 2 3'.split())
    assert type(args.foo) == list
    assert args.foo == [1, 2, 3]


def test_argument_list_dm_options():
    al = ArgumentList([DmOptions()])
    args = al.parse_args('--dm-options foo=1=int'.split())
    assert type(args.dm_options) == dict
    assert args.dm_options['foo'] == 1


def test_argument_list_dm_commands():
    al = ArgumentList([DmCommands()])
    args = al.parse_args('--dm-commands init fit-fixed set-scale_var'.split())
    assert type(args.dm_commands) == list
    assert args.dm_commands[0] == 'init'
    assert args.dm_commands[1] == 'fit fixed'
    assert args.dm_commands[2] == 'set scale_var'


def test_options():
    options = ['foo=hello=str', 'bar=1=int', 'foobar=1.0=float']
    option_dict = parse_options(options)
    assert len(option_dict) == 3
    assert option_dict['foo'] == 'hello'
    assert option_dict['bar'] == 1
    assert option_dict['foobar'] == 1.0


def test_encode_options():
    options = {'foo': 'hello', 'bar': 1, 'foobar': 1.0}
    encoded = encode_options(options)
    assert encoded[0] == 'foo=hello=str'
    assert encoded[1] == 'bar=1=int'
    assert encoded[2] == 'foobar=1.0=float'


def test_invalid_encode_options():
    options = {'foo': np.array([1, 2])}
    with pytest.raises(OptionParsingError):
        encode_options(options)


def test_invalid_parse_options():
    options = ['foo=[1, 2]=list']
    with pytest.raises(OptionParsingError):
        parse_options(options)


def test_parse_commands():
    assert parse_commands(['init']) == ['init']
    assert parse_commands(['init', 'fit-fixed']) == ['init', 'fit fixed']


def test_encode_commands():
    commands = ['init', 'fit fixed', 'set scale_var']
    new = encode_commands(commands)
    assert new[0] == 'init'
    assert new[1] == 'fit-fixed'
    assert new[2] == 'set-scale_var'
