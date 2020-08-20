from cascade_at.executor.args.args import _Argument, IntArg, StrArg, FloatArg
from cascade_at.executor.args.args import ModelVersionID, LogLevel, ParentLocationID, SexID


def test_argument():
    a = _Argument()
    assert a._flag is None
    assert type(a._parser_kwargs) == dict

    a = _Argument('--arg')
    assert a._flag == '--arg'


def test_int_arg():
    a = IntArg()
    assert a._parser_kwargs['type'] == int


def test_str_arg():
    a = StrArg()
    assert a._parser_kwargs['type'] == str


def test_float_arg():
    a = FloatArg()
    assert a._parser_kwargs['type'] == float


def test_model_version_id():
    a = ModelVersionID()
    assert a._flag == '--model-version-id'
    assert a._parser_kwargs['type'] == int
    assert a._parser_kwargs['required']


def test_log_level():
    a = LogLevel()
    assert a._flag == '--log-level'
    assert a._parser_kwargs['type'] == str
    assert not a._parser_kwargs['required']


def test_parent_location_id():
    a = ParentLocationID()
    assert a._flag == '--parent-location-id'
    assert a._parser_kwargs['type'] == int
    assert a._parser_kwargs['required']


def test_sex_id():
    a = SexID()
    assert a._flag == '--sex-id'
    assert a._parser_kwargs['type'] == int
    assert a._parser_kwargs['required']
