import string

import pytest

from hypothesis import given
import hypothesis.strategies as st

from cascade.core.parameters import _ParameterHierarchy, ParameterProperty

valid_attribute_names = st.text(string.ascii_letters + string.digits + "_", min_size=1)


@st.composite
def parameter_dicts(draw, attribute_names):
    max_size = 10
    return draw(
        st.dictionaries(
            attribute_names,
            st.recursive(
                st.none() | st.booleans() | st.integers() | st.text(string.printable),
                lambda children: st.dictionaries(attribute_names, children, max_size=max_size),
                max_leaves=10,
            ),
            max_size=max_size,
        )
    )


@given(parameters=parameter_dicts(valid_attribute_names))
def test_ParameterHierarchy__basic_initialization(parameters):
    ph = _ParameterHierarchy(**parameters)

    def _validate(param_hier, src_dict):
        for k, v in src_dict.items():
            if isinstance(v, dict):
                _validate(getattr(param_hier, k), v)
            else:
                assert getattr(param_hier, k) == v

    _validate(ph, parameters)


def test_ParameterHierarchy__freezing():
    ph = _ParameterHierarchy(a="a", b="b")
    ph._frozen = True

    with pytest.raises(TypeError):
        ph.a = "c"

    with pytest.raises(TypeError):
        # You shouldn't be able to un-freeze it
        ph._frozen = False


@pytest.fixture
def parameter_container():
    class TestParameters:
        thing = ParameterProperty()
        other_thing = ParameterProperty()

    return TestParameters()


def testParameterProperty__set(parameter_container):
    parameter_container.thing = {"a": "a", "b": "b"}


def testParameterProperty__get(parameter_container):
    parameter_container.thing = {"a": "a", "b": "b", "c": {"d": "d"}}
    assert parameter_container.thing.a == "a"
    assert parameter_container.thing.b == "b"
    assert parameter_container.thing.c.d == "d"


def testParameterProperty__get_missing(parameter_container):
    with pytest.raises(AttributeError):
        parameter_container.thing
