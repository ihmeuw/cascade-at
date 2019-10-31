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
from cascade_at.inputs.utilities.gbd_ids import get_location_set_version_id
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

        :param model_version_id: (int)
        :param gbd_round_id: (int)
        :param decomp_step_id: (int)
        :param csmr_process_version_id: (int) process version ID for CSMR
        :param csmr_cause_id: (int) cause to pull CSMR from
        :param crosswalk_version_id: (int) crosswalk version to use
        :param country_covariate_id: (list of int) list of covariate IDs
        :param conn_def: (str)
        :param location_set_version_id: (int) can be None, if it's none, get the
            best location_set_version_id for estimation hierarchy of this GBD round.
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
        self.data = None
        self.covariates = None
        self.location_dag = None
        self.age_groups = None

        self.asdr_for_dismod = None
        self.csmr_for_dismod = None
        self.data_for_dismod = None
        self.all_dismod_data = None
        self.covariate_data = None
        self.country_covariate_data = None
        self.country_covariate_specs = None

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

    def configure_inputs_for_dismod(self, settings):
        """
        Modifies the inputs for DisMod based on model-specific settings.
        :param settings: (cascade.settings.configuration.Configuration)
        :return: self
        """
        self.data_for_dismod = self.data.configure_for_dismod(
            measures_to_exclude=self.measures_to_exclude_from_settings(settings),
            data_eta=self.data_eta_from_settings(settings),
            density=self.density_from_settings(settings),
            nu=self.nu_from_settings(settings)
        )
        self.asdr_for_dismod = self.asdr.configure_for_dismod()
        self.csmr_for_dismod = self.csmr.configure_for_dismod()
        self.all_dismod_data = pd.concat([
            self.data_for_dismod,
            self.asdr_for_dismod,
            self.csmr_for_dismod
        ], axis=0)
        self.country_covariate_data = {c.covariate_id: c.configure_for_dismod() for c in self.covariate_data}
        self.country_covariate_specs = CovariateSpecs(settings.country_covariate)

        return self

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
        The default data eta is np.nan
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
        :param settings: (cascade.settings.configuration.Configuration)
        :return:
        """
        nu = defaultdict(lambda: np.nan)
        nu["students"] = settings.students_dof.data
        nu["log_students"] = settings.log_students_dof.data
        return nu
