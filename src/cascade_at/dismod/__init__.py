from cascade_at.core import CascadeATError


class DismodATException(CascadeATError):
    """This means DismodAT complained. Could be bad data or a model problem.

    It's difficult to figure out which errors are bad input and which
    are solution problems. This error is for all text output from
    DismodAT that complains.
    """
