import numpy as np
import pandas as pd
from collections import defaultdict

from gbd.decomp_step import decomp_step_from_decomp_step_id

from cascade_at.core.log import get_loggers
from cascade_at.inputs.asdr import ASDR
from cascade_at.inputs.csmr import CSMR
from cascade_at.inputs.covariate_data import CovariateData
from cascade_at.inputs.covariate_specs import CovariateSpecs
from cascade_at.inputs.data import CrosswalkVersion
from cascade_at.inputs.demographics import Demographics
from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.population import Population
from cascade_at.inputs.utilities.covariate_weighting import get_interpolated_covariate_values
from cascade_at.inputs.utilities.gbd_ids import get_location_set_version_id, SEX_MAP
from cascade_at.dismod.integrand_mappings import make_integrand_map

LOG = get_loggers(__name__)


class Inputs:
    def __init__(self, model_version_id, gbd_round_id,
                 decomp_step_id, csmr_process_version_id,
                 csmr_cause_id, crosswalk_version_id,
                 country_covariate_id,
                 conn_def,
                 location_set_version_id=None):
        """
        The class that constructs all of the inputs.

        Parameters:
            model_version_id: (int) the model version ID
            gbd_round_id: (int) the GBD round ID
            decomp_step_id: (int) the decomp step ID
            csmr_process_version_id: (int) process version ID for CSMR
            csmr_cause_id: (int) cause to pull CSMR from
            crosswalk_version_id: (int) crosswalk version to use
            country_covariate_id: (list of int) list of covariate IDs
            conn_def: (str) connection definition from .odbc file (e.g. 'epi')
            location_set_version_id: (int) can be None, if it's none, get the
                best location_set_version_id for estimation hierarchy of this GBD round.

        Attributes:
            decomp_step: (str) the decomp step in string form
            demographics: (cascade_at.inputs.demographics.Demographics) a demographics object
                that specifies the age group, sex, location, and year IDs to grab
            integrand_map: (dict) dictionary mapping from GBD measure IDs to DisMod IDs
            asdr: (cascade_at.inputs.asdr.ASDR) all-cause mortality input object
            csmr: (cascade_at.inputs.csmr.CSMR) cause-specific mortality input object from cause
                csmr_cause_id
            data: (cascade_at.inputs.data.CrosswalkVersion) crosswalk version data from IHME database
            covariate_data: (List[cascade_at.inputs.covariate_data.CovariateData]) list of covariate
                data objects that contains the raw covariate data mapped to IDs
            location_dag: (cascade_at.inputs.locations.LocationDAG) DAG of locations to be used
            population: (cascade_at.inputs.population.Population) population object that is used
                for covariate weighting
            data_eta: (Dict[str, float]): dictionary of eta value to be applied to each measure
            density: (Dict[str, str]): dictionary of density to be applied to each measure
            nu: (Dict[str, float]): dictionary of nu value to be applied to each measure
            dismod_data: (pd.DataFrame) resulting dismod data formatted to be used in the dismod database
        
        Usage:
        >>> from cascade_at.settings.base_case import BASE_CASE
        >>> from cascade_at.settings.settings import load_settings

        >>> settings = load_settings(BASE_CASE)
        >>> covariate_ids = [i.country_covariate_id for i in settings.country_covariate]

        >>> i = Inputs(model_version_id=settings.model.model_version_id,
        >>>            gbd_round_id=settings.gbd_round_id,
        >>>            decomp_step_id=settings.model.decomp_step_id,
        >>>            csmr_process_version_id=None,
        >>>            crosswalk_version_id=settings.model.crosswalk_version_id,
        >>>            country_covariate_id=settings.covariate_ids,
        >>>            conn_def='epi',
        >>>            location_set_version_id=settings.location_set_version_id)
        >>> i.get_raw_inputs()
        >>> i.configure_inputs_for_dismod()
        """
        self.model_version_id = model_version_id
        self.gbd_round_id = gbd_round_id
        self.decomp_step_id = decomp_step_id
        self.csmr_process_version_id = csmr_process_version_id
        self.csmr_cause_id = csmr_cause_id
        self.crosswalk_version_id = crosswalk_version_id
        self.country_covariate_id = country_covariate_id
        self.conn_def = conn_def

        self.decomp_step = decomp_step_from_decomp_step_id(
            self.decomp_step_id
        )

        self.demographics = Demographics(gbd_round_id=self.gbd_round_id)

        if location_set_version_id is None:
            self.location_set_version_id = get_location_set_version_id(
                gbd_round_id=self.gbd_round_id
            )
        else:
            self.location_set_version_id = location_set_version_id

        self.integrand_map = make_integrand_map()

        self.exclude_outliers = True
        self.asdr = None
        self.csmr = None
        self.population = None
        self.data = None
        self.covariates = None
        self.location_dag = None
        self.age_groups = None

        self.data_eta = None
        self.density = None
        self.nu = None
        self.measures_to_exclude = None

        self.dismod_data = None
        self.covariate_data = None
        self.country_covariate_data = None
        self.covariate_specs = None

    def get_raw_inputs(self):
        """
        Get the raw inputs that need to be used
        in the modeling.

        :return:
        """
        LOG.info("Getting all raw inputs.")
        self.asdr = ASDR(
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw()
        self.csmr = CSMR(
            cause_id=self.csmr_cause_id,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id,
            process_version_id=self.csmr_process_version_id
        ).get_raw()
        self.data = CrosswalkVersion(
            crosswalk_version_id=self.crosswalk_version_id,
            exclude_outliers=self.exclude_outliers,
            demographics=self.demographics,
            conn_def=self.conn_def
        ).get_raw()
        self.covariate_data = [CovariateData(
            covariate_id=c,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw() for c in self.country_covariate_id]
        self.location_dag = LocationDAG(
            location_set_version_id=self.location_set_version_id
        )
        self.population = Population(
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_population()

    def configure_inputs_for_dismod(self, settings):
        """
        Modifies the inputs for DisMod based on model-specific settings.

        :param settings: (cascade.settings.configuration.Configuration)
        :return: self
        """
        self.data_eta = self.data_eta_from_settings(settings)
        self.density = self.density_from_settings(settings)
        self.nu = self.nu_from_settings(settings)
        self.measures_to_exclude = self.measures_to_exclude_from_settings(settings)

        self.dismod_data = pd.concat([
            self.data.configure_for_dismod(measures_to_exclude=self.measures_to_exclude),
            self.asdr.configure_for_dismod(),
            self.csmr.configure_for_dismod()
        ], axis=0)

        self.dismod_data["density"] = self.dismod_data.measure.apply(self.density.__getitem__)
        self.dismod_data["eta"] = self.dismod_data.measure.apply(self.data_eta.__getitem__)
        self.dismod_data["nu"] = self.dismod_data.measure.apply(self.nu.__getitem__)

        self.dismod_data.reset_index(drop=True, inplace=True)

        self.country_covariate_data = {c.covariate_id: c.configure_for_dismod() for c in self.covariate_data}
        self.covariate_specs = CovariateSpecs(settings.country_covariate)

        self.interpolate_country_covariate_values()

        self.dismod_data.drop(['age_group_id'], inplace=True, axis=1)
        self.dismod_data['sex'] = self.dismod_data.sex_id.map(SEX_MAP)

        return self

    def interpolate_country_covariate_values(self):
        """
        Interpolates the covariate values onto the data
        so that the non-standard ages and years match up to meaningful
        covariate values.
        """
        for c in self.covariate_specs.covariate_specs:
            if c.study_country == 'country':
                LOG.info(f"Interpolating and merging the country covariate {c.covariate_id}.")
                cov_df = self.country_covariate_data[c.covariate_id]
                interpolated_mean_value = get_interpolated_covariate_values(
                    data_df=self.dismod_data,
                    covariate_df=cov_df,
                    population_df=self.population.raw
                )
                self.dismod_data[c.name] = interpolated_mean_value

    def measures_to_exclude_from_settings(self, settings):
        """
        Gets the measures to exclude from the data from the model
        settings configuration.
        :param settings: (cascade.settings.configuration.Configuration)
        :return:
        """
        if not settings.model.is_field_unset("exclude_data_for_param"):
            measures_to_exclude = [self.integrand_map[m].name
                                   for m in settings.model.exclude_data_for_param
                                   if m in self.integrand_map]
        else:
            measures_to_exclude = list()
        if settings.policies.exclude_relative_risk:
            measures_to_exclude.append("relrisk")
        return measures_to_exclude

    def data_eta_from_settings(self, settings):
        """
        Gets the data eta from the settings Configuration.
        The default data eta is np.nan.
        settings.eta.data: (Dict[str, float]): Default value for eta parameter on distributions
            as a dictionary from measure name to float

        :param settings: (cascade.settings.configuration.Configuration)
        :return:
        """
        data_eta = defaultdict(lambda: np.nan)
        if not settings.eta.is_field_unset("data") and settings.eta.data:
            data_eta = defaultdict(lambda: float(settings.eta.data))
        for set_eta in settings.data_eta_by_integrand:
            data_eta[self.integrand_map[set_eta.integrand_measure_id]] = float(set_eta.value)
        return data_eta

    def density_from_settings(self, settings):
        """
        Gets the density from the settings Configuration.
        The default density is "gaussian".
        settings.model.data_density: (Dict[str, float]): Default values for density parameter on distributions
            as a dictionary from measure name to string

        :param settings: (cascade.settings.configuration.Configuration)
        :return:
        """
        density = defaultdict(lambda: "gaussian")
        if not settings.model.is_field_unset("data_density") and settings.model.data_density:
            density = defaultdict(lambda: settings.model.data_density)
        for set_density in settings.data_density_by_integrand:
            density[self.integrand_map[set_density.integrand_measure_id]] = set_density.value
        return density

    @staticmethod
    def nu_from_settings(settings):
        """
        Gets nu from the settings Configuration.
        The default nu is np.nan.
        settings.students_dof.data: (Dict[str, float]): The parameter for students-t distributions
        settings.log_students_dof.data: (Dict[str, float]): The parameter for students-t distributions in log-space

        :param settings: (cascade.settings.configuration.Configuration)
        :return:
        """
        nu = defaultdict(lambda: np.nan)
        nu["students"] = settings.students_dof.data
        nu["log_students"] = settings.log_students_dof.data
        return nu
