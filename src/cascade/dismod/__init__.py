from cascade.core import CascadeError


class DismodATException(CascadeError):
    """This means DismodAT complained. Could be bad data or a model problem.

    It's difficult to figure out which errors are bad input and which
    are solution problems. This error is for all text output from
    DismodAT that complains.
    """
