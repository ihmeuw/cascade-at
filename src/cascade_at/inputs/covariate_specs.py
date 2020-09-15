from typing import List

from cascade_at.inputs.utilities.covariate_specifications import create_covariate_specifications
from cascade_at.model.covariate import Covariate
from cascade_at.settings.settings_config import CountryCovariate, StudyCovariate
from cascade_at.inputs.utilities.gbd_ids import get_study_level_covariate_ids, get_country_level_covariate_ids
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


class CovariateSpecs:
    def __init__(self, country_covariates: List[CountryCovariate], study_covariates: List[StudyCovariate]):

        self.covariate_list = []
        self.country_covariates = country_covariates
        self.study_covariates = study_covariates
        self.covariate_multipliers, self.covariate_specs = create_covariate_specifications(
            country_covariate=self.country_covariates,
            study_covariate=self.study_covariates
        )

        self.country_covariate_ids = {
            spec.covariate_id for spec in self.covariate_specs
            if spec.study_country == "country"
        }
        self.study_id_to_name = get_study_level_covariate_ids()
        self.country_id_to_name = get_country_level_covariate_ids(list(self.country_covariate_ids))

        for cov in self.covariate_specs:
            if cov.study_country == 'study':
                short = self.study_id_to_name.get(cov.covariate_id, None)
            elif cov.study_country == 'country':
                short = self.country_id_to_name.get(cov.covariate_id, None)
            else:
                raise RuntimeError("Must be either study or country covariates.")
            if short is None:
                raise RuntimeError(f"Covariate {cov} is not found in id-to-name mapping.")
            cov.untransformed_covariate_name = short
    
    def create_covariate_list(self):
        """
        Creates a list of Covariate objects with the current reference value and max difference.
        """
        self.covariate_list = []
        for c in self.covariate_specs:
            self.covariate_list.append(Covariate(c.name, c.reference, c.max_difference))
        return self
