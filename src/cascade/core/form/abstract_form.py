class NoValue:
    def __repr__(self):
        return "NO_VALUE"


NO_VALUE = NoValue()


class Node:
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


class Field(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate_and_normalize(self, instance, ignore_missing_keys=False):
        value = self.__get__(instance, instance.__class__)
        if value is NO_VALUE:
            if not ignore_missing_keys and not self.nullable:
                return [("", "Missing data")]
            return []

        new_value, error = self._validate_and_normalize(instance, value)
        if error is not None:
            return [("", error)]
        self.__set__(instance, new_value)
        return []

    def _validate_and_normalize(self, instance, value):
        return value, None


class SimpleTypeField(Field):
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

    def new_instance(self):
        return self.__class__(*self._args, **self._kwargs)

    def process_source(self, source):
        if source:
            for k, v in source.items():
                for c in self._children:
                    if c.name == k:
                        setattr(self, k, v)
                        break

        if self.name_field:
            setattr(self, self.name_field, self.name)

    def validate_and_normalize(self, instance=None, ignore_missing_keys=False):
        errors = []
        ignore_missing_keys = ignore_missing_keys or self.nullable
        for child, child_value in self._child_instances.items():
            if isinstance(child_value, Node):
                child = child_value
            c_errors = child.validate_and_normalize(self, ignore_missing_keys)
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
        return errors
