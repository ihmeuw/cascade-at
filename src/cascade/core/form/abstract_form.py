""" This module defines general tools for building validators for messy
hierarchical parameter data. It tries to follow conventions from form
validation systems in the web application world since that is a very similar
problem.

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


class NoValue:
    """Represents an unset value, which is distinct from None because None may
    actually appear in input data.
    """

    def __repr__(self):
        return "NO_VALUE"


NO_VALUE = NoValue()


class Node:
    """ Base class for all form components.
    """

    _children = None

    def __init__(self, nullable=False):
        self.nullable = nullable
        self.name = None
        if self._children:
            self._child_instances = {c: NO_VALUE for c in self._children}
        else:
            self._child_instances = {}

    def __get__(self, instance, owner):
        if isinstance(instance, Node):
            return instance._child_instances[self]
        else:
            return None

    def is_unset(self, instance):
        value = self.__get__(instance, type(self))
        if value is NO_VALUE:
            return True
        return False

    def __set__(self, instance, value):
        if isinstance(instance, Node):
            instance._child_instances[self] = value

    def __set_name__(self, owner, name):
        if issubclass(owner, Node):
            if owner._children is None:
                owner._children = {self}
            else:
                owner._children.add(self)
        self.name = name

    @property
    def unset(self):
        value = self.__class__.__get__(self, self.__class__)
        if value is NO_VALUE:
            return True
        return False


class Field(Node):
    """ A field within a form. Fields are responsible for validating the data
    they contain (without respect to data in other fields) and transforming
    it into a normalized form.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate_and_normalize(self, instance):
        """ Validates the data for this field on the given parent instance and
        transforms the data into it's normalized form. The actual details of
        validating and transforming are delegated to subclasses except for
        checking for missing data which is handled here.

        Args:
            instance (Form): the instance of the form for which this field
                             should be validated.

        Returns:
            [(str, str)]: a list of error messages with path strings
                          showing where in this object they occured. For most
                          fields the path will always be empty.
        """
        if self.is_unset(instance):
            if not self.nullable:
                return [("", "Missing data")]
            return []

        value = self.__get__(instance, type(instance))
        new_value, error = self._validate_and_normalize(instance, value)
        if error is not None:
            return [("", error)]
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
                          showing where in this object they occured. For most
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
                     ValueError or TypeError if transformation is not
                     possible.
    """

    def __init__(self, constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constructor = constructor

    def _validate_and_normalize(self, instance, value):
        try:
            new_value = self.constructor(value)
        except (ValueError, TypeError):
            return None, f"Invalid {self.constructor.__name__} value '{value}'"
        return new_value, None


class Form(Node):
    """ The parent class of all forms.

    Args:
        source (dict): The input data to parse. If None, it can be supplied
                       later by calling process_source
        name_field (str): If supplied then a field of the same name must be
                          present on the subclass. That field will always have
                          the name of the attribute this class is assigned to
                          in it's parent rather than the value, if any, that
                          the field had in the input data.
    """

    def __init__(self, source=None, name_field=None, nullable=False):
        super().__init__(nullable)
        self._args = []
        self._kwargs = {"name_field": name_field, "nullable": nullable}
        self.name_field = name_field
        if source:
            self.process_source(source)

    def __set__(self, instance, value):
        if isinstance(instance, Node):
            form = self.new_instance()
            form.name = self.name
            form.process_source(value)
            instance._child_instances[self] = form

    @property
    def children(self):
        for child, child_value in self._child_instances.items():
            if isinstance(child_value, Node):
                child = child_value
            yield child

    def is_unset(self, instance=None):
        return all([c.is_unset(self) for c in self.children])

    def new_instance(self):
        return type(self)(*self._args, **self._kwargs)

    def process_source(self, source):
        if source:
            for k, v in source.items():
                for c in self._children:
                    if c.name == k:
                        setattr(self, k, v)
                        break

        if self.name_field:
            setattr(self, self.name_field, self.name)

    def validate_and_normalize(self, instance=None):
        if self.is_unset() and self.nullable:
            return []

        errors = []
        for child in self.children:
            c_errors = child.validate_and_normalize(self)
            if c_errors:
                if child.name:
                    # TODO: This replace is ugly and probably means that I'm
                    # not thinking clearly about how these error paths
                    # get constructed.
                    errors.extend(
                        [((f"{child.name}." + p).replace(".[", "[") if p else child.name, e) for p, e in c_errors]
                    )
                else:
                    errors.extend(c_errors)
        if not errors:
            errors = [("", e) for e in self._full_form_validation(instance)]

        return errors

    @property
    def unset(self):
        return all([c.unset for c in self.children])

    def _full_form_validation(self, instance):
        return []
