"""
Represents covariates in the model.
"""
import logging


CODELOG = logging.getLogger(__name__)
MATHLOG = logging.getLogger(__name__)


class CovariateColumns:
    """
    Establishes a reference value for a covariate column on input data
    and in output data.

    Args:
        column_name (str): Name of hte column in the input data.
        reference (float): Required reference where covariate has no effect.
        max_difference (float, optional):
            If a data point's covariate is farther than `max_difference`
            from the reference value, then this data point is excluded
            from the calculation.
    """
    def __init__(self, column_name, reference, max_difference=None):
        self.name = column_name
        self.reference = reference
        self.max_difference = max_difference


class CovariateMultiplier:
    """
    A covariate multiplier makes a covariate column predict a response
    on its target.
    """
    def __init__(self, covariate_column, smooth):
        """
        Args:
            covariate_column (CovariateColumn): Which predictor to use.
            smooth (Smooth): Each covariate gets a smoothing grid.
        """
        self.column = covariate_column
        self.smooth = smooth
