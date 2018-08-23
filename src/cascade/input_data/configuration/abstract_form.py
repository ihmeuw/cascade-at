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


class _Form(_Attribute):
    _fields = set()

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
                raise ConfigurationError(f"Extra key '{key}' in form '{self._name}'")

        if not self.ignore_missing_keys:
            missing = self._fields.difference(processed_fields)
            if missing:
                raise ConfigurationError(f"Missing keys for form '{self._name}': {missing}")

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        if self._name_field:
            setattr(self, self._name_field, name)

    def _validate(self):
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
        print(self._name, value)
        obj._field_values[self._name] = value

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        owner._fields.add(name)

    def _validate_and_normalize(self, value):
        raise NotImplementedError()


class IntegerField(_Field):
    def _validate_and_normalize(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid integer value '{value}' for {self._name}")


class FloatField(_Field):
    def _validate_and_normalize(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid integer value '{value}' for {self._name}")


class BooleanField(_Field):
    def _validate_and_normalize(self, value):
        if isinstance(value, bool):
            return value
        elif value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise ConfigurationError(f"Invalid boolean value: {value} for {self._name}")


class StringField(_Field):
    def _validate_and_normalize(self, value):
        try:
            return str(value)
        except (ValueError, TypeError):
            raise ConfigurationError(f"Invalid string value: {value} for {self._name}")


class OptionField(_Field):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options

    def _validate_and_normalize(self, value):
        if value in self.options:
            return value
        else:
            raise ConfigurationError(f"Invalid option '{value}' must be one of {self.options} for {self._name}")


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
