""" This module defines general tools for building validators for messy
hierarchical parameter data. It provides a declarative API for creating form
validators. It tries to follow conventions from form validation systems in the
web application world since that is a very similar problem.

Example:
    Validators are defined as classes with attributes which corrispond to the
    values they expect to receive. For example, consider this JSON blob:

    {"field_a": "10", "field_b": "22.4", "nested": {"field_c": "Some Text"}}

    A validator for that document would look like this:

    class NestedValidator(Form):
        field_c = SimpleTypeField(str)

    class BlobValidator(Form):
        field_a = SimpleTypeField(int)
        field_b = SimpleTypeField(int)
        nested = NestedValidator()

    And could be used as follows:

    >>> form = BlobValidator(json.loads(document))
    >>> form.validate_and_normalize()
    >>> form.field_a
    10
    >>> form.nested.field_c
    "Some Text"
"""

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class NoValue:
    """Represents an unset value, which is distinct from None because None may
    actually appear in input data.
    """

    def __repr__(self):
        return "NO_VALUE"


NO_VALUE = NoValue()


class FormComponent:
    """ Base class for all form components. It bundles up behavior shared by
    both (sub)Forms and Fields.

    Note:
        FormComponent, Form and Field all make heavy use of the descriptor
        protocol (https://docs.python.org/3/howto/descriptor.html). That means
        that the relationship between objects and the data they operate on is
        more complex than usual. Read up on descriptors, if you aren't familiar,
        and pay close attention to how __set__ and __get__ access data.

    Args:
        nullable (bool): If False then missing data for this node is considered
          an error. Defaults to False.
        default: Default value to return if unset
        name (str): The name used in the EpiViz interface.
    """

    _children = None

    def __init__(self, nullable=False, default=None, display=None):
        self._nullable = nullable
        self._default = default
        self._name = None
        self._display_name = display
        if self._children:
            self._child_instances = {c: NO_VALUE for c in self._children}
        else:
            self._child_instances = {}

    def __get__(self, instance, owner):
        if isinstance(instance, FormComponent):
            value = instance._child_instances[self]
            if value is NO_VALUE and self._nullable:
                return self._default
            return value
        else:
            return None

    @property
    def display_name(self):
        return self._display_name if self._display_name else self._name

    def is_unset(self, instance):
        value = instance._child_instances[self]
        if value is NO_VALUE:
            return True
        return False

    def __set__(self, instance, value):
        if isinstance(instance, FormComponent):
            instance._child_instances[self] = value

    def __set_name__(self, owner, name):
        if issubclass(owner, FormComponent):
            if owner._children is None:
                owner._children = {self}
            else:
                owner._children.add(self)
        self._name = name

    @property
    def unset(self):
        value = self.__class__.__get__(self, self.__class__)
        if value is NO_VALUE:
            return True
        return False


class Field(FormComponent):
    """ A field within a form. Fields are responsible for validating the data
    they contain (without respect to data in other fields) and transforming
    it into a normalized form.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate_and_normalize(self, instance, root=None):
        """ Validates the data for this field on the given parent instance and
        transforms the data into it's normalized form. The actual details of
        validating and transforming are delegated to subclasses except for
        checking for missing data which is handled here.

        Args:
            instance (Form): the instance of the form for which this field
                             should be validated.

        Returns:
            [(str, str, str)]: a list of error messages with path strings
                          showing where in this object they occurred. For most
                          fields the path will always be empty.
        """
        if self.is_unset(instance):
            if not self._nullable:
                return [("", "", "Missing data")]
            return []

        value = self.__get__(instance, type(instance))
        new_value, error = self._validate_and_normalize(instance, value)
        if error is not None:
            return [("", "", error)]
        self.__set__(instance, new_value)
        return []

    def _validate_and_normalize(self, instance, value):
        """ Validation and normalization details to be handled in overridden
        methods in subclasses.

        Args:
            instance (Form): the instance of the form for which this field
                             should be validated.
            value: The value of this field on the parent instance.

        Returns:
            [(str, str)]: a list of error messages with path strings
                          showing where in this object they occurred. For most
                          fields the path will always be empty.
        """
        return value, None


class SimpleTypeField(Field):
    """A field which transforms input data using a constructor function and
    emits errors if that transformation fails.

    In general this is used to convert to simple types like int or float.
    Because it emits only very simple messages it is not appropriate for
    cases where the cause of any error isn't obvious from knowing the
    name of the constructor function and a string representation of the
    input value.

    Args:
        constructor: a function which takes one argument and returns a
                     normalized version of that argument. It must raise
                     ValueError, TypeError or OverflowError if transformation
                     is not possible.
    """

    def __init__(self, constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constructor = constructor

    def _validate_and_normalize(self, instance, value):
        try:
            new_value = self.constructor(value)
        except (ValueError, TypeError, OverflowError):
            return None, f"Invalid {self.constructor.__name__} value '{value}'"
        return new_value, None


class Form(FormComponent):
    """ The parent class of all forms.

    Validation for forms happens in two stages. First all the form's fields and
    sub forms are validated. If none of those have errors, then the form is
    known to be in a consistent state and it's `_full_form_validation` method
    is run to finalize validation. If any field or sub form is invalid then
    this form's `_full_form_validation` method will not be run because the form
    may be in an inconsistent state.

    Simple forms will be valid if all their fields are valid but more complex
    forms will require additional checks across multiple fields which are
    handled by `_full_form_validation`.

    Note:
      A nested form may be marked nullable. It is considered null if all of
      it's children are null. If a nullable form is null then it is not an
      error for non-nullable fields in it to be null. If any of the form's
      fields are non-null then the whole form is considered non-null at which
      point missing data for non-nullable fields becomes an error again.

    Args:
        source (dict): The input data to parse. If None, it can be supplied
                       later by calling process_source
        name_field (str): If supplied then a field of the same name must be
                          present on the subclass. That field will always have
                          the name of the attribute this class is assigned to
                          in it's parent rather than the value, if any, that
                          the field had in the input data.
    """

    def __init__(self, source=None, name_field=None, nullable=False, display=None):
        super().__init__(nullable=nullable, display=display)
        self._args = []
        self._kwargs = {"name_field": name_field, "nullable": nullable}
        self._name_field = name_field
        if source:
            self.process_source(source)

    def __set__(self, instance, value):
        if isinstance(instance, FormComponent):
            form = self.new_instance()
            form._name = self._name
            form._display_name = self._display_name
            form.process_source(value)
            instance._child_instances[self] = form

    @property
    def children(self):
        for child, child_value in self._child_instances.items():
            if isinstance(child_value, FormComponent):
                child = child_value
            yield child

    def is_field_unset(self, field_name):
        child = type(self).__dict__[field_name]
        child_value = self._child_instances[child]
        if isinstance(child_value, FormComponent):
            child = child_value
        return child.is_unset(self)

    def is_unset(self, instance=None):
        return all([c.is_unset(self) for c in self.children])

    def new_instance(self):
        return type(self)(*self._args, **self._kwargs)

    def process_source(self, source):
        if source:
            for k, v in source.items():
                for c in self._children:
                    if c._name == k:
                        setattr(self, k, v)
                        break

        if self._name_field:
            setattr(self, self._name_field, self._name)

    def validate_and_normalize(self, instance=None, root=None):
        if self.is_unset() and self._nullable:
            return []

        errors = []

        if root is None:
            root = self

        for child in self.children:
            c_errors = child.validate_and_normalize(self, root=root)
            if c_errors:
                if child._name:
                    # TODO: This replace is ugly and probably means that I'm
                    # not thinking clearly about how these error paths
                    # get constructed.
                    errors.extend(
                        [(
                            (f"{child._name}." + p).replace(".[", "[") if p else child._name,
                            (f"{child.display_name}." + h).replace(".[", "[") if h else child.display_name,
                            e
                        )
                            for p, h, e in c_errors]
                    )
                else:
                    errors.extend(c_errors)
        if not errors:
            errors = [("", "", e) for e in self._full_form_validation(root)]

        return errors

    @property
    def unset(self):
        return all([c.unset for c in self.children])

    def _full_form_validation(self, root):
        """ Can be overridden by subclasses to do any validation that requires
        multiple fields. This method will only execute if all the form's fields
        are themselves valid and have been normalize. If not overridden it does
        no additional checks.
        """
        return []
