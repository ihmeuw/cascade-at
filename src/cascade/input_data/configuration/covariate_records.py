"""
This is responsible for

*  Making covariate records consistent between study and country covariates
*  Setting them up to be put on the model.
"""
import pandas as pd

from cascade.core.log import getLoggers

CODELOG, MATHLOG = getLoggers(__name__)


class CovariateRecords:
    """
    Data structure to store covariate data that is part of the bundle but not
    yet put on the model.
    """
    __slots__ = ["kind", "measurements", "average_integrand_cases", "id_to_name",
                 "id_to_reference", "id_to_max_difference"]

    def __init__(self, study_or_country):
        self.kind = study_or_country
        self.measurements = pd.DataFrame()
        """Pandas dataframe where each column is the id of a covariate
        and the index matches the bundle index"""

        self.average_integrand_cases = pd.DataFrame()
        """Pandas dataframe where each column is the id of a covariate
        and the index matches the avgint index."""

        self.id_to_name = {}
        """Dict from string name to integer id."""

        self.id_to_reference = {}
        """The reference value for this covariate."""

        self.id_to_max_difference = {}
        """If a value is farther than this from reference, it is excluded."""
