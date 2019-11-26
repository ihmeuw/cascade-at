from cascade_at.context.arg_utils import parse_options, parse_commands


def test_options():
    options = ['foo=hello=str', 'bar=1=int', 'foobar=1.0=float']
    option_dict = parse_options(options)
    assert len(option_dict) == 3
    assert option_dict['foo'] == 'hello'
    assert option_dict['bar'] == 1
    assert option_dict['foobar'] == 1.0


def test_commands():
    assert parse_commands(['init']) == ['init']
    assert parse_commands(['init', 'fit-fixed']) == ['init', 'fit fixed']

