from functools import total_ordering
from itertools import chain

from cascade.input_data.configuration.id_map import make_integrand_map, PRIMARY_INTEGRANDS_TO_RATES
from cascade.input_data.configuration.builder import COVARIATE_TRANSFORMS


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
        return (self.study_country, self.covariate_id, self.transformation_id)

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
        group_name = dict(rate_value="alpha", meas_value="beta", meas_std="gamma")
        return group_name[self.grid_spec.mulcov_type]

    @property
    def key(self):
        """Key for the :py:class:`DismodGroups` object, so it is a tuple
        of (covariate name, rate) or (covariate name, integrand) where rate
        and integrand are strings."""
        id_to_integrand = make_integrand_map()
        if self.group == "alpha":
            rate_measure = id_to_integrand[self.grid_spec.measure_id].name
            measure = PRIMARY_INTEGRANDS_TO_RATES[rate_measure]
        else:
            measure = id_to_integrand[self.grid_spec.measure_id].name
        return (self.covariate.name, measure)

    def __repr__(self):
        return f"EpiVizCovariateMultiplier{self.covariate.spec}"


def kind_and_id(covariate_setting):
    for kind in ["study", "country"]:
        if hasattr(covariate_setting, f"{kind}_covariate_id"):
            return kind, getattr(covariate_setting, f"{kind}_covariate_id")
    raise AssertionError(
        f"Covariate should be either study or country {covariate_setting}"
    )


def create_covariate_specifications(study, country):
    """Parses EpiViz-AT settings to create two data structures for Covariate creation.

    Args:
        study (List[Form]): The study_covariate member of the EpiViz-AT settings.
        country (List[Form]): The country_covariate member of the EpiViz-AT settings.

    Returns:
         (List[EpiVizCovariateMultiplier], List[EpiVizCovariate]): The multipliers
         are specification for making SmoothGrids. The covariates are specification
         for downloading data and attaching it to the bundle and average integrand
         tables. The multipliers use the covariates in order to know the name
         of the covariate.
    """
    covariate_specs = set()
    for setting in chain(study, country):
        # This tells us what the data is for the column.
        kind, covariate_id = kind_and_id(setting)
        covariate_specs.add((kind, covariate_id, setting.transformation))
    covariate_specs.add(("study", 0, 0))  # Sex covariate
    covariate_specs.add(("study", 1604, 0))  # One covariate
    covariate_dict = {cspec: EpiVizCovariate(*cspec) for cspec in covariate_specs}

    multipliers = list()
    for setting in chain(study, country):
        kind, covariate_id = kind_and_id(setting)
        # This tells us what the data is for the column.
        covariate_id = getattr(setting, f"{kind}_covariate_id")
        spec = (kind, covariate_id, setting.transformation)
        multipliers.append(EpiVizCovariateMultiplier(covariate_dict[spec], setting))

    return multipliers, sorted(covariate_dict.values())
