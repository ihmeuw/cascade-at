import numpy as np
import pytest

from cascade_at.context.arg_utils import parse_options, parse_commands
from cascade_at.context.arg_utils import encode_options, encode_commands, OptionParsingError


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

