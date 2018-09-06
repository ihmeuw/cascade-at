from cascade.core.form.abstract_form import Form, Field, SimpleTypeField


def test_development_target():
    class IntField(Field):
        def _validate_and_normalize(self, instance, value):
            try:
                new_value = int(value)
            except (ValueError, TypeError):
                return None, f"Invalid integer value '{value}'"
            return new_value, None

    class MyInnerForm(Form):
        my_inner_field = IntField()

    class MyForm(Form):
        my_field = IntField()
        my_inner_form = MyInnerForm()

    f = MyForm({"my_field": "10", "my_inner_form": {"my_inner_field": 100}})
    errors = f.validate_and_normalize()
    assert not errors

    assert f.my_field == 10
    assert f.my_inner_form.my_inner_field == 100

    f = MyForm({"my_field": "10", "my_inner_form": {"my_inner_field": "eaoeao"}})
    errors = f.validate_and_normalize()
    assert set(errors) == {("my_inner_form.my_inner_field", "Invalid integer value 'eaoeao'")}

    f = MyForm({"my_field": None, "my_inner_form": {}})
    errors = f.validate_and_normalize()
    assert set(errors) == {
        ("my_field", "Invalid integer value 'None'"),
        ("my_inner_form.my_inner_field", "Missing data"),
    }

    errors = f.validate_and_normalize(ignore_missing_keys=True)
    assert set(errors) == {("my_field", "Invalid integer value 'None'")}


def test_Form__name_field():
    class MyInnerForm(Form):
        my_field = SimpleTypeField(int)
        foo = SimpleTypeField(str)

    class MyOuter(Form):
        inner_form = MyInnerForm(name_field="foo")

    f = MyOuter({"inner_form": {"my_field": "10", "foo": "not_inner_form"}})
    assert not f.validate_and_normalize()
    assert f.inner_form.foo == "inner_form"


def test_SimpleTypeField__validation():
    class MyForm(Form):
        my_int_field = SimpleTypeField(int)
        my_float_field = SimpleTypeField(float)

    f = MyForm({"my_int_field": 10, "my_float_field": 1.5})
    errors = f.validate_and_normalize()
    assert not errors

    f = MyForm({"my_int_field": "oueou", "my_float_field": "blaa"})
    errors = f.validate_and_normalize()
    assert set(errors) == {
        ("my_int_field", "Invalid int value 'oueou'"),
        ("my_float_field", "Invalid float value 'blaa'"),
    }


def test_SimpleTypeField__normalization():
    class MyForm(Form):
        my_int_field = SimpleTypeField(int)
        my_float_field = SimpleTypeField(float)

    f = MyForm({"my_int_field": 10, "my_float_field": 1.5})
    f.validate_and_normalize()
    assert f.my_int_field == 10
    assert f.my_float_field == 1.5
