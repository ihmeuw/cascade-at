from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


class _ParameterHierarchy:
    """An immutable store which gives dot notation access to hierarchical key-value data
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = _ParameterHierarchy(**value)
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if hasattr(self, "_frozen") and self._frozen:
            raise TypeError("'_ParameterHierarchy' object does not support assignment")
        super().__setattr__(key, value)


class ParameterProperty:
    """A property like object which converts dictionaries into _ParameterHierarchy objects on assignment.
    """

    def __init__(self):
        self._name = None
        self._underlying_attribute = None

    def __set_name__(self, owner, name):
        self._name = name
        self._underlying_attribute = f"_parameters_{name}"

    def __get__(self, instance, owner):
        if not hasattr(instance, self._underlying_attribute):
            raise AttributeError(f"Parameters '{self._name}' not yet set")

        return getattr(instance, self._underlying_attribute)

    def __set__(self, instance, parameters):
        parameters = _ParameterHierarchy(**parameters)
        setattr(instance, self._underlying_attribute, parameters)
