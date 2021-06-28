from functools import total_ordering
from itertools import chain
from typing import List

from cascade_at.dismod.integrand_mappings import make_integrand_map, PRIMARY_INTEGRANDS_TO_RATES
from cascade_at.inputs.utilities.transformations import COVARIATE_TRANSFORMS
from cascade_at.settings.settings_config import StudyCovariate, CountryCovariate


@total_ordering
class EpiVizCovariate:
    """This specifies covariate data from settings.
    It is separate from the cascade.model.Covariate, which is a Dismod-AT
    covariate. EpiViz-AT distinguishes study and country covariates and
    encodes them into the Dismod-AT covariate names.
    """
    def __init__(self, study_country, covariate_id, transformation_id):
        self.study_country = study_country
        self.covariate_id = covariate_id
        self.transformation_id = transformation_id
        """Which function to apply to this covariate column (log, exp, etc)"""
        self.untransformed_covariate_name = None
        """The name for this covariate before transformation."""
        self.reference = 0
        self.max_difference = None

    @property
    def spec(self):
        """Unique identifier for a covariate because two multipliers may
        refer to the same covariate."""
        return self.study_country, self.covariate_id, self.transformation_id

    @property
    def name(self):
        """The name for this covariate in the final data."""
        if self.untransformed_covariate_name is None:
            raise RuntimeError(
                f"The name for this covariate hasn't been set yet "
                f"id={self.covariate_id}, {self.study_country}, "
                f"transform={self.transformation_id}."
            )
        transform_name = COVARIATE_TRANSFORMS[self.transformation_id].__name__
        if transform_name != "identity":
            return f"{self.untransformed_covariate_name}_{transform_name}"
        else:
            return f"{self.untransformed_covariate_name}"

    def __eq__(self, other):
        return self.spec == other.spec

    def __hash__(self):
        return hash(self.spec)

    def __lt__(self, other):
        """Comparison method to ensure sex is first, and one is second."""
        if not isinstance(other, EpiVizCovariate):
            return NotImplemented
        sex = ("study", 0, 0)
        one = ("study", 1604, 0)
        a, b = (self.spec, other.spec)
        if a == sex or (a == one and b != sex):
            return True
        elif a == one and b == sex:
            return False
        else:
            return self.spec < other.spec

    def __repr__(self):
        return f"EpiVizCovariate{self.spec}"


class EpiVizCovariateMultiplier:
    """Used to build priors for this covariate multiplier."""
    def __init__(self, covariate, settings):
        """
        Args:
            covariate (EpiVizCovariate): The covariate
            settings (StudyCovariate|CountryCovariate): Section of the form.
        """
        self.covariate = covariate  # Keep a link in order to find its name later.
        self.grid_spec = settings

    @property
    def group(self):
        """The name of the DismodGroups group, so it's alpha, beta, or gamma."""
        group_name = dict(rate_value="alpha", meas_value="beta", meas_noise="gamma")
        return group_name[self.grid_spec.mulcov_type]

    @property
    def key(self):
        """Key for the :py:class:`DismodGroups` object, so it is a tuple
        of (covariate name, rate) or (covariate name, integrand) where rate
        and integrand are strings."""
        id_to_integrand = make_integrand_map()
        if self.group == "alpha":
            if self.grid_spec.measure_id in ['pini', 'iota', 'rho', 'chi', 'omega']:
                # New EpiVis returns rate names for the rate covariates.
                measure = self.grid_spec.measure_id
            else:
                # Old EpiVis returns of integrand measure id's for rate covariates.
                rate_measure = id_to_integrand[int(self.grid_spec.measure_id)].name
                measure = PRIMARY_INTEGRANDS_TO_RATES[rate_measure]
        else:
            measure = id_to_integrand[int(self.grid_spec.measure_id)].name
        return self.covariate.name, measure

    def __repr__(self):
        return f"EpiVizCovariateMultiplier{self.covariate.spec}"


def kind_and_id(covariate_setting):
    for kind in ["study", "country"]:
        if hasattr(covariate_setting, f"{kind}_covariate_id"):
            return kind, getattr(covariate_setting, f"{kind}_covariate_id")
    raise AssertionError(
        f"Covariate should be either study or country {covariate_setting}"
    )


def create_covariate_specifications(country_covariate: List[CountryCovariate],
                                    study_covariate: List[StudyCovariate]) -> (List[EpiVizCovariateMultiplier],
                                                                     List[EpiVizCovariate]):
    """Parses EpiViz-AT settings to create two data structures for Covariate creation.

    Covariate multipliers will only contain country covariates.
    Covariate specifications will contain both the country and study covariates,
    which are only the 'sex' and 'one' covariates.

    >>> from cascade_at.settings.base_case import BASE_CASE
    >>> from cascade_at.settings.settings import load_settings
    >>> settings = load_settings(BASE_CASE)
    >>> multipliers, data_spec = create_covariate_specifications(settings.country_covariate, settings.study_covariate)

    Parameters
    ----------
    country_covariate
        The country_covariate member of the EpiViz-AT settings.
    study_covariate
        The study_covariate member of the EpiViz-AT settings.

    Returns
    -------
     The multipliers are specification for making SmoothGrids.
     The covariates are specification
     for downloading data and attaching it to the crosswalk version and average integrand
     tables. The multipliers use the covariates in order to know the name
     of the covariate.
    """
    covariate_specs = set()

    if study_covariate is None:
        study_covariate = []
    if country_covariate is None:
        country_covariate = []
    for setting in chain(study_covariate, country_covariate):
        # This tells us what the data is for the column.
        kind, covariate_id = kind_and_id(setting)
        covariate_specs.add((kind, covariate_id, setting.transformation))
    covariate_specs.add(("study", 0, 0))  # Sex covariate
    covariate_specs.add(("study", 1604, 0))  # One covariate
    covariate_dict = {cspec: EpiVizCovariate(*cspec) for cspec in covariate_specs}

    multipliers = list()
    for setting in chain(study_covariate, country_covariate):
        kind, covariate_id = kind_and_id(setting)
        # This tells us what the data is for the column.
        covariate_id = getattr(setting, f"{kind}_covariate_id")
        spec = (kind, covariate_id, setting.transformation)
        multipliers.append(EpiVizCovariateMultiplier(covariate_dict[spec], setting))

    return multipliers, sorted(covariate_dict.values())
