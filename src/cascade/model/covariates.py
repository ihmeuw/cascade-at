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
    """
    def __init__(self):
        self._data = None

    def add(self, column_name, reference, max_difference=None):
        self.column_name = column_name
        self.reference = reference
        self.max_difference = max_difference


class CovariateMultiplier:
    """
    A covariate multiplier makes a covariate column predict a response
    on its target.
    """
    def __init__(self, covariate_column, target, smooth_grid):
        """

        Args:
            covariate_column (CovariateColumn): Which predictor to use.
            target (str):
                The name of a primary rate or integrand. Specified as
                two words, where the first is a rate or integrand
                and the second is value or stdev, such as
                "iota value" or "prevalence value" or "remission stddev".

            smooth_grid (PriorGrid): Each covariate gets a smoothing grid.
        """
        self._column = covariate_column
        self.target = target.split()
        self.smooth_grid = smooth_grid


