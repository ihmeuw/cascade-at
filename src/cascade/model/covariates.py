"""
Represents covariates in the model.
"""
from numbers import Number
from numpy import isnan

from cascade.core import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class Covariate:
    """
    Establishes a reference value for a covariate column on input data
    and in output data. It is possible to create a covariate column with
    nothing but a name, but it must have a reference value before it
    can be used in a model.

    Args:
        column_name (str): Name of hte column in the input data.
        reference (float, optional):
            Reference where covariate has no effect.
        max_difference (float, optional):
            If a data point's covariate is farther than `max_difference`
            from the reference value, then this data point is excluded
            from the calculation. Must be greater than or equal to zero.
    """
    def __init__(self, column_name, reference=None, max_difference=None):
        self._name = None
        self._reference = None
        self._max_difference = None

        self.name = column_name
        if reference is not None:
            self.reference = reference
        self.max_difference = max_difference

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, nom):
        if not isinstance(nom, str):
            raise TypeError(f"Covariate name must be a string, not {nom}")
        if len(nom) < 1:
            raise ValueError(f"Covariate name must not be empty string")
        self._name = nom

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, ref):
        self._reference = float(ref)

    @property
    def max_difference(self):
        return self._max_difference

    @max_difference.setter
    def max_difference(self, difference):
        if difference is None or isinstance(difference, Number) and isnan(difference):
            self._max_difference = None
        else:
            diff = float(difference)
            if diff < 0:
                raise ValueError(
                    f"max difference for a covariate must be greater than "
                    f"or equal to zero, not {difference}")
            self._max_difference = diff

    def __hash__(self):
        return hash((self._name, self._reference, self._max_difference))

    def __repr__(self):
        return f"Covariate({self.name}, {self.reference}, {self.max_difference})"

    def __eq__(self, other):
        if not isinstance(other, Covariate):
            raise NotImplementedError(f"Cannot compare a covariate and a {type(other)}: {other}.")
        return (self._name == other.name and self._reference == other._reference
                and self._max_difference == other._max_difference)
