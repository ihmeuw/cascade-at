import pytest

from cascade.core.form import Form, FormList, IntField, OptionField, StringListField


@pytest.fixture
def form_with_list():
    class MyInnerForm(Form):
        int_field = IntField()

    class MyOuterForm(Form):
        inner_forms = FormList(MyInnerForm)

    return MyOuterForm


@pytest.fixture
def form_with_options():
    class MyForm(Form):
        str_option_field = OptionField(["a", "b", "see"])
        int_option_field = OptionField([1, 2, 3], constructor=int)

    return MyForm


@pytest.fixture
def form_with_string_list():
    class MyForm(Form):
        ints_field = StringListField(seperator=" ", constructor=int)

    return MyForm


def form_with_listList__empty(form_with_list):
    f = form_with_list({"inner_forms": []})
    assert not f.validate()
    assert len(f.inner_forms) == 0


def form_with_listList__non_empty(form_with_list):
    f = form_with_list({"inner_forms": [{"int_field": "10"}, {"int_field": "20"}]})
    assert not f.validate()
    assert f.inner_forms[0].int_field == "10"
    assert f.inner_forms[1].int_field == "20"


def form_with_listList__validation(form_with_list):
    f = form_with_list({"inner_forms": [{"int_field": "oeueoueo"}, {"int_field": "20"}]})
    assert f.validate() == [("inner_forms[0].int_field", "Invalid int value 'oeueoueo'")]


def test_OptionField__success(form_with_options):
    f = form_with_options({"str_option_field": "a", "int_option_field": "1"})
    assert not f.validate()


def test_OptionField__validation(form_with_options):
    f = form_with_options({"str_option_field": "c", "int_option_field": "oueo"})
    assert set(f.validate()) == {
        ("str_option_field", "Invalid option 'c'"),
        ("int_option_field", "Invalid int value 'oueo'"),
    }


def test_OptionField__normalization(form_with_options):
    f = form_with_options({"str_option_field": "a", "int_option_field": "3"})
    f.normalize()
    assert f.str_option_field == "a"
    assert f.int_option_field == 3


def test_StringListField__success(form_with_string_list):
    f = form_with_string_list({"ints_field": "1 2 3 4 5"})
    assert not f.validate()


def test_StringListField__validation(form_with_string_list):
    f = form_with_string_list({"ints_field": "1 2 three 4 five"})
    assert f.validate() == [("ints_field", "Errors in items: [Invalid int value 'three', Invalid int value 'five']")]
