from cascade_at.core import CascadeATError


class ModelError(CascadeATError):
    """A problem with setup or solution of model.

    It's a RuntimeError.
    """
