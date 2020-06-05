from cascade_at.model.covariates import Covariate
from cascade_at.model.dismod_groups import DismodGroups
from cascade_at.model.model import Model
from cascade_at.model.object_wrapper import ObjectWrapper
from cascade_at.model.priors import (
    Uniform, Constant, Gaussian, Laplace, StudentsT, LogGaussian, LogLaplace, LogStudentsT
)
from cascade_at.model.session import Session
from cascade_at.model.smooth_grid import SmoothGrid
from cascade_at.model.var import Var
from cascade_at.core import CascadeError


class ModelError(CascadeError):
    """A problem with setup or solution of model.

    It's a RuntimeError.
    """


__all__ = [
    Model, Session, DismodGroups, SmoothGrid, Var, Covariate, ObjectWrapper,
    Uniform, Constant, Gaussian, Laplace, StudentsT, LogGaussian, LogLaplace,
    LogStudentsT, ModelError
]
