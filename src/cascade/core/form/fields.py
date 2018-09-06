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

    def validate(self, instance=None, ignore_missing_keys=False):
        return [(f"[{i}].{p}", e) for i, form in enumerate(self) for (p, e) in form.validate(self, ignore_missing_keys)]

    def normalize(self, instance=None):
        for form in self:
            form.normalize(self)

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

    def _validate(self, instance, value):
        if self.constructor:
            try:
                value = self.constructor(value)
            except (ValueError, TypeError):
                return f"Invalid {self.constructor.__name__} value '{value}'"

        if value not in self.options:
            return f"Invalid option '{value}'"
        return None

    def _normalize(self, instance, value):
        if self.constructor:
            return self.constructor(value)
        return value


class StringListField(SimpleTypeField):
    def __init__(self, *args, constructor=str, seperator=" ", **kwargs):
        super().__init__(constructor, *args, **kwargs)
        self.seperator = seperator

    def _validate(self, instance, value):
        errors = []
        for item in value.split(self.seperator):
            error = super()._validate(instance, item)
            if error:
                errors.append(error)
        if errors:
            return "Errors in items: [" + ", ".join(errors) + "]"

    def _normalize(self, instance, value):
        parent_normalize = super()._normalize
        return [parent_normalize(instance, item) for item in value.split(self.seperator)]
