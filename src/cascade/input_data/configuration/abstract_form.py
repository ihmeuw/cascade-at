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
        if owner._fields is None:
            owner._fields = set()
        owner._fields.add(self)

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

    @property
    def null(self):
        return True


class _Form(_Attribute):
    _fields = None

    def __init__(self, *args, name_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._name_field = name_field
        self._field_values = {}
        self.initialized = False

    def load_json(self, source_json, path="", ignore_extra_keys=False):
        self._process_json(source_json, path, ignore_extra_keys)
        self.initialized = True

    def _process_json(self, source_json, path, ignore_extra_keys):
        processed_fields = set()
        for key, value in source_json.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), _Form):
                    getattr(self, key).load_json(value, path, ignore_extra_keys)
                else:
                    setattr(self, key, value)
                processed_fields.add(key)
            elif not self.ignore_extra_keys:
                raise ConfigurationError(f"Extra key '{key}' in form '{self}'")

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        if self._name_field:
            setattr(self, self._name_field, name)

    def validate_and_normalize(self, parent=None, path="", ignore_missing_keys=False):
        path += "." + self._name
        errors = []
        fields = self._fields or []
        if not ignore_missing_keys:
            missing = {f._name for f in fields if isinstance(f, _Field)}.difference(self._field_values.keys())
            missing |= {f._name for f in fields if isinstance(f, _Form) and not f.initialized}
            if missing:
                errors.append((path, f"Missing keys for form '{self}': {missing}"))
        for field in fields:
            errors.extend(field.validate_and_normalize(self, path, ignore_missing_keys))
        return errors

    @property
    def null(self):
        for field in self._fields:
            if not field.null:
                return False
        return True


class DummyForm(_Form):
    def _process_json(self, source_json, path, ignore_extra_keys):
        pass


class FormList(_Form):
    def __init__(self, form_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.form_class = form_class
        self.items = []

    def _process_json(self, source_json, path, ignore_extra_keys):
        items = []
        for i, item in enumerate(source_json):
            form = self.form_class()
            form._owner = self
            form._name = ""
            form.load_json(item, path)
            items.append(form)
        self.items = items

    def validate_and_normalize(self, parent=None, path="", ignore_missing_keys=False):
        errors = []
        for i, form in enumerate(self):
            errors.extend(form.validate_and_normalize(self, path + f".{self._name}[{i}]", ignore_missing_keys))
        return errors

    def __iter__(self):
        return iter(self.items)


class _Field(_Attribute):
    def __init__(self, *args, nullable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nullable = nullable
        self._value = None

    def __get__(self, obj, owner=None):
        return obj._field_values.get(self._name)

    def __set__(self, obj, value):
        obj._field_values[self._name] = value

    def validate_and_normalize(self, parent, path, ignore_missing_keys=False):
        value = self.__get__(parent)
        # FIXME The viz code uses empty strings as nulls. We should change this as soon as they fix that.
        # https://jira.ihme.washington.edu/browse/EPI-998
        if not (self.nullable and (value is None or value == "")):
            value, errors = self._validate_and_normalize(value, path)
            if errors:
                return errors
        else:
            value = None
        self.__set__(parent, value)
        return []

    def _validate_and_normalize(self, value):
        raise NotImplementedError()

    @property
    def null(self):
        pass


class IntegerField(_Field):
    def _validate_and_normalize(self, value, path):
        try:
            return int(value), None
        except (ValueError, TypeError):
            return None, [(path + "." + self._name, f"Invalid integer value '{value}' for {self}")]


class FloatField(_Field):
    def _validate_and_normalize(self, value, path):
        try:
            return float(value), None
        except (ValueError, TypeError):
            return None, [(path + "." + self._name, f"Invalid float value '{value}' for {self}")]


class BooleanField(_Field):
    def _validate_and_normalize(self, value, path):
        if isinstance(value, bool):
            return value, None
        elif value == 0:
            return False, None
        elif value == 1:
            return True, None
        else:
            return None, [(path + "." + self._name, f"Invalid boolean value '{value}' for {self}")]


class StringField(_Field):
    def _validate_and_normalize(self, value, path):
        try:
            return str(value), None
        except (ValueError, TypeError):
            return None, [(path + "." + self._name, f"Invalid string value '{value}' for {self}")]


class StringAsListField(_Field):
    def __init__(self, seperator, inner_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seperator = seperator
        self.inner_type = inner_type

    def _validate_and_normalize(self, value, path):
        try:
            value = str(value)
        except (ValueError, TypeError):
            return None, [(path + "." + self._name, f"Invalid string value '{value}' for {self}")]

        values = value.split(self.seperator)
        errors = []
        if self.inner_type:
            converted_values = []
            for i, v in enumerate(values):
                try:
                    converted_values.append(self.inner_type(v))
                except (ValueError, TypeError):
                    errors.append(
                        (
                            path + f".{self._name}[{i}]",
                            f"Invalid {self.inner_type} value: {v} for element {i} of {self}",
                        )
                    )
            values = converted_values
        return values, errors


class OptionField(_Field):
    def __init__(self, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options

    def _validate_and_normalize(self, value, path):
        if value in self.options:
            return value, None
        else:
            return (
                None,
                [(path + "." + self._name, f"Invalid option '{value}' must be one of {self.options} for {self}")],
            )
