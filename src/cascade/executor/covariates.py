from functools import lru_cache, total_ordering
from itertools import chain

from cascade.core.db import db_queries
from cascade.input_data.configuration.id_map import make_integrand_map, PRIMARY_INTEGRANDS_TO_RATES


@lru_cache(maxsize=1)
def country_covariate_names():
    """Returns a dictionary from ``covariate_id`` to covariate short name."""
    covariate_df = db_queries.get_ids("covariate")[["covariate_id", "covariate_name_short"]].set_index("covariate_id")
    return covariate_df.to_dict()["covariate_name_short"]


class CovariateMultiplier:
    """Used to build priors for this covariate multiplier."""
    def __init__(self, covariate, settings):
        """
        Args:
            covariate (EpiVizCovariate): The covariate
            settings (StudyCovariate|CountryCovariate): Section of the form.
        """
        self.covariate = covariate
        self.grid_spec = settings

    @property
    def group(self):
        group_name = dict(rate_value="alpha", meas_value="beta", meas_std="gamma")
        return group_name[self.grid_spec.mulcov_type]

    @property
    def key(self):
        """Key for the :py:class:`DismodGroups` object."""
        id_to_integrand = make_integrand_map()
        if self.group == "alpha":
            rate_measure = id_to_integrand[self.grid_spec.measure_id].name
            measure = PRIMARY_INTEGRANDS_TO_RATES[rate_measure]
        else:
            measure = id_to_integrand[self.grid_spec.measure_id].name
        return (self.covariate.name, measure)


@total_ordering
class EpiVizCovariate:
    """This is related to the covariate data."""
    def __init__(self, study_country, covariate_id, transformation_id):
        self.study_country = study_country
        self.covariate_id = covariate_id
        self.transformation_id = transformation_id
        self.name = None

    @property
    def spec(self):
        return (self.study_country, self.covariate_id, self.transformation_id)

    def __eq__(self, other):
        return self.spec == other.spec

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


def create_covariate_specifications(study, country):
    both_lists = list(chain((("study", s) for s in study), (("country", c) for c in country)))
    covariate_specs = set()
    for kind, setting in both_lists:
        # This tells us what the data is for the column.
        covariate_id = getattr(setting, f"{kind}_covariate_id")
        covariate_specs.add((kind, covariate_id, setting.transformation))
    covariate_dict = {cspec: EpiVizCovariate(*cspec) for cspec in covariate_specs}

    multipliers = list()
    for kind, setting in both_lists:
        # This tells us what the data is for the column.
        covariate_id = getattr(setting, f"{kind}_covariate_id")
        spec = (kind, covariate_id, setting.transformation)
        multipliers.append(CovariateMultiplier(covariate_dict[spec], setting))

    return multipliers, sorted(covariate_dict.values())
