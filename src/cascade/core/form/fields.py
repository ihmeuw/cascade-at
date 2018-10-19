""" This module defines specializations of the general tools in abstract_form,
mostly useful field types.
"""
from cascade.core.form.abstract_form import Form, Field, SimpleTypeField, NO_VALUE

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class IntField(SimpleTypeField):
    def __init__(self, *args, **kwargs):
        super().__init__(int, *args, **kwargs)


class FloatField(SimpleTypeField):
    def __init__(self, *args, **kwargs):
        super().__init__(float, *args, **kwargs)


class StrField(SimpleTypeField):
    def __init__(self, *args, **kwargs):
        super().__init__(str, *args, **kwargs)


class FormList(Form):
    """ This represents a homogeneous list of forms. For example, it might be
    used to contain a list of priors within a smoothing grid.

    Args:
      inner_form_constructor: A factory which produces an instance of a Form
      subclass. Most often it will just be the Form subclass itself.
    """

    def __init__(self, inner_form_constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner_form_constructor = inner_form_constructor
        self._forms = []
        self._args = [inner_form_constructor] + self._args

    def process_source(self, source):
        forms = []
        for i, inner_source in enumerate(source):
            form = self._inner_form_constructor()
            form._name = str(i)
            form.process_source(inner_source)
            forms.append(form)
        self._forms = forms

    def validate_and_normalize(self, instance=None, root=None):
        return [
            (f"[{i}].{p}", f"[{i}].{h}", e) for i, form in enumerate(self) for (p, h, e) in form.validate_and_normalize(self, root=root)
        ]

    def __get__(self, instance, owner):
        value = super().__get__(instance, owner)
        if value is NO_VALUE:
            return []
        return value

    def __iter__(self):
        return iter(self._forms)

    def __len__(self):
        return len(self._forms)

    def __getitem__(self, key):
        return self._forms[key]


class Dummy(Field):
    """ A black hole which consumes all values without error. Use to mark
    sections of the configuration which have yet to be implemented and should
    be ignored.
    """

    def validate_and_normalize(self, instance, root=None):
        return []

    def process_source(self, source):
        pass

    def __get__(self, instance, owner):
        return NO_VALUE

    def __set__(self, instance, value):
        pass


class OptionField(SimpleTypeField):
    """ A field which will only accept values form a predefined set.

    Args:
        options (list): The list of options to choose from
        constructor: A function which takes a string and returns the expected
          type. Behaves as the constructor for SimpleTypeField. Defaults to str
    """

    def __init__(self, options, *args, constructor=str, **kwargs):
        super().__init__(constructor, *args, **kwargs)
        self.options = options

    def _validate_and_normalize(self, instance, value):
        new_value, error = super()._validate_and_normalize(instance, value)
        if error:
            return None, error

        if new_value not in self.options:
            return None, f"Invalid option '{new_value}'"

        return new_value, None


class StringListField(SimpleTypeField):
    """ A field which takes a string containing values demarcated by some
    separator and transforms them into a homogeneous list of items of an
    expected type.

    Args:
        constructor: A function which takes a string and returns the expected
          type. Behaves as the constructor for SimpleTypeField. Defaults to str
        separator (str): The string to split by. Defaults to a single space.
    """

    def __init__(self, *args, constructor=str, separator=" ", **kwargs):
        super().__init__(constructor, *args, **kwargs)
        self.separator = separator

    def _validate_and_normalize(self, instance, value):
        errors = []
        new_values = []

        if isinstance(value, str):
            values = value.split(self.separator)
        else:
            # This case hits when there's only a single numerical value in the
            # list because Epiviz switches from strings to the actual numerical
            # type in that case.
            values = [value]

        for item in values:
            new_value, error = super()._validate_and_normalize(instance, item)
            if error:
                errors.append(error)
            else:
                new_values.append(new_value)
        if errors:
            return None, "Errors in items: [" + ", ".join(errors) + "]"
        return new_values, None
