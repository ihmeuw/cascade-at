from cascade.model.covariates import Covariate
from cascade.model.dismod_groups import DismodGroups
from cascade.model.model import Model
from cascade.model.object_wrapper import ObjectWrapper
from cascade.model.priors import (
    Uniform, Constant, Gaussian, Laplace, StudentsT, LogGaussian, LogLaplace, LogStudentsT
)
from cascade.model.session import Session
from cascade.model.smooth_grid import SmoothGrid
from cascade.model.var import Var
from cascade.core import CascadeError


class ModelError(CascadeError):
    """A problem with setup or solution of model.

    It's a RuntimeError.
    """


__all__ = [
    Model, Session, DismodGroups, SmoothGrid, Var, Covariate, ObjectWrapper,
    Uniform, Constant, Gaussian, Laplace, StudentsT, LogGaussian, LogLaplace,
    LogStudentsT, ModelError
]
