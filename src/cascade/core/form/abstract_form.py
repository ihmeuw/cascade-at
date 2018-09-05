NO_VALUE = object()


class Node:
    _children = None

    def __init__(self):
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
    def __init__(self, *args, nullable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nullable = nullable

    def validate(self, instance, ignore_missing_keys=False):
        value = self.__get__(instance, instance.__class__)
        if value is NO_VALUE:
            if not ignore_missing_keys:
                return [("", "Missing data")]
            return []

        error = self._validate(instance, value)
        if error is not None:
            return [("", error)]
        return []

    def _validate(self, instance, value):
        raise NotImplementedError()

    def normalize(self, instance):
        value = self.__get__(instance, instance.__class__)
        value = self._normalize(instance, value)
        self.__set__(instance, value)

    def _normalize(self, instance, value):
        raise NotImplementedError()


class SimpleTypeField(Field):
    def __init__(self, constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constructor = constructor

    def _validate(self, instance, value):
        try:
            self.constructor(value)
        except (ValueError, TypeError):
            return f"Invalid {self.constructor.__name__} value '{value}'"
        return None

    def _normalize(self, instance, value):
        return self.constructor(value)


class Form(Node):
    def __init__(self, source=None, name_field=None):
        super().__init__()
        self._args = []
        self._kwargs = {"name_field": name_field}
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
        if self.name_field:
            setattr(self, self.name_field, self.name)
        for k, v in source.items():
            for c in self._children:
                if c.name == k:
                    setattr(self, k, v)
                    break

    def validate(self, instance=None, ignore_missing_keys=False):
        errors = []
        for child, child_value in self._child_instances.items():
            if isinstance(child_value, Node):
                child = child_value
            c_errors = child.validate(self, ignore_missing_keys)
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

    def normalize(self, instance=None):
        for child, child_value in self._child_instances.items():
            if isinstance(child_value, Node):
                child = child_value
            child.normalize(self)
