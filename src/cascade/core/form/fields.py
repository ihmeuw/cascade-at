from cascade.core.form.abstract_form import Form, Field, SimpleTypeField


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
    def __init__(self, inner_form_constructor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_form_constructor = inner_form_constructor
        self._forms = []
        self._args = [inner_form_constructor] + self._args

    def process_source(self, source):
        forms = []
        for i, inner_source in enumerate(source):
            form = self.inner_form_constructor()
            form.name = str(i)
            form.process_source(inner_source)
            forms.append(form)
        self._forms = forms

    def validate_and_normalize(self, instance=None, ignore_missing_keys=False):
        return [
            (f"[{i}].{p}", e)
            for i, form in enumerate(self)
            for (p, e) in form.validate_and_normalize(self, ignore_missing_keys)
        ]

    def __iter__(self):
        return iter(self._forms)

    def __len__(self):
        return len(self._forms)

    def __getitem__(self, key):
        return self._forms[key]


class Dummy(Field):
    def validate(self, instance, ignore_missing_keys=False):
        return []

    def normalize(self, instance):
        pass

    def process_source(self, source):
        pass


class OptionField(Field):
    def __init__(self, options, *args, constructor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options
        self.constructor = constructor

    def _validate_and_normalize(self, instance, value):
        new_value = value
        if self.constructor:
            try:
                new_value = self.constructor(value)
            except (ValueError, TypeError):
                return None, f"Invalid {self.constructor.__name__} value '{value}'"

        if new_value not in self.options:
            return None, f"Invalid option '{new_value}'"
        return new_value, None


class StringListField(SimpleTypeField):
    def __init__(self, *args, constructor=str, seperator=" ", **kwargs):
        super().__init__(constructor, *args, **kwargs)
        self.seperator = seperator

    def _validate_and_normalize(self, instance, value):
        errors = []
        new_values = []
        for item in value.split(self.seperator):
            new_value, error = super()._validate_and_normalize(instance, item)
            if error:
                errors.append(error)
            else:
                new_values.append(new_value)
        if errors:
            return None, "Errors in items: [" + ", ".join(errors) + "]"
        return new_values, None
