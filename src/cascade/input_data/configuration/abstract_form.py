from cascade.input_data.configuration import ConfigurationError


class _Attribute:
    def __init__(self):
        self._owner = None
        self._name = ""
        # FIXME: This is a stupid system for controlling the handling of
        # missingness
        self._ignore_extra_keys = None
        self._ignore_missing_keys = None

    def __set_name__(self, owner, name):
        self._owner = owner
        self._name = name

    @property
    def ignore_extra_keys(self):
        if self._ignore_extra_keys is None:
            if isinstance(self._owner, _Attribute) or (
                isinstance(self._owner, type) and issubclass(self._owner, _Attribute)
            ):
                return self._owner.ignore_extra_keys
            else:
                return False
        return self._ignore_extra_keys

    @ignore_extra_keys.setter
    def ignore_extra_keys(self, value):
        self._ignore_extra_keys = value

    @property
    def ignore_missing_keys(self):
        if self._ignore_missing_keys is None:
            if isinstance(self._owner, _Attribute) or (
                isinstance(self._owner, type) and issubclass(self._owner, _Attribute)
            ):
                return self._owner.ignore_missing_keys
            else:
                return False
        return self._ignore_missing_keys

    @ignore_missing_keys.setter
    def ignore_missing_keys(self, value):
        self._ignore_missing_keys = value

    def __repr__(self):
        cname = self.__class__.__name__
        return f"<{cname} {self._name}>"


class _Form(_Attribute):
    _fields = None

    def __init__(self, *args, name_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._name_field = name_field
        self._field_values = {}

    def load_json(self, source_json):
        self._process_json(source_json)
        self._validate()

    def _process_json(self, source_json):
        processed_fields = set()
        for key, value in source_json.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), _Form):
                    getattr(self, key).load_json(value)
                else:
                    setattr(self, key, value)
                processed_fields.add(key)
            elif not self.ignore_extra_keys:
                raise ConfigurationError(f"Extra key '{key}' in form '{self}'")

        if not self.ignore_missing_keys:
            missing = self._fields.difference(processed_fields)
            if missing:
                raise ConfigurationError(f"Missing keys for form '{self}': {missing}")

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        if self._name_field:
            setattr(self, self._name_field, name)

    def _validate(self):
        pass


class DummyForm(_Form):
    def _process_json(self, source_json):
        pass


class _Field(_Attribute):
    def __init__(self, *args, nullable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nullable = nullable
        self._value = None

    def __get__(self, obj, owner=None):
        return obj._field_values.get(self._name)

    def __set__(self, obj, value):
        # FIXME The viz code uses empty strings as nulls. We should change this as soon as they fix that.
        if not (self.nullable and (value is None or value == "")):
            value = self._validate_and_normalize(value)
        else:
            value = None
        obj._field_values[self._name] = value

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        if owner._fields is None:
            owner._fields = set()
        owner._fields.add(name)

    def _validate_and_normalize(self, value):
        raise NotImplementedError()


class IntegerField(_Field):
    def _validate_and_normalize(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid integer value '{value}' for {self}")


class FloatField(_Field):
    def _validate_and_normalize(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid float value '{value}' for {self}")


class BooleanField(_Field):
    def _validate_and_normalize(self, value):
        if isinstance(value, bool):
            return value
        elif value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise ConfigurationError(f"Invalid boolean value: {value} for {self}")


class StringField(_Field):
    def _validate_and_normalize(self, value):
        try:
            return str(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid string value: {value} for {self}")


class StringAsListField(_Field):
    def __init__(self, seperator, inner_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seperator = seperator
        self.inner_type = inner_type

    def _validate_and_normalize(self, value):
        try:
            value = str(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid string value: {value} for {self}")

        values = value.split(self.seperator)
        if self.inner_type:
            converted_values = []
            for i, v in enumerate(values):
                try:
                    converted_values.append(self.inner_type(v))
                except (ValueError, TypeError):
                    raise ConfigurationError(f"Invalid {self.inner_type} value: {v} for element {i} of {self}")
            values = converted_values
        return values


class OptionField(_Field):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options

    def _validate_and_normalize(self, value):
        if value in self.options:
            return value
        else:
            raise ConfigurationError(f"Invalid option '{value}' must be one of {self.options} for {self}")


class FormList(_Field):
    def __init__(self, form_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.form_class = form_class

    def _validate_and_normalize(self, value):
        items = []
        for i, item in enumerate(value):
            form = self.form_class()
            form._owner = self
            form._name = str(i)
            form.load_json(item)
            items.append(form)
        return items
