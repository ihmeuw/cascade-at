import numpy as np

from gbd.decomp_step import decomp_step_from_decomp_step_id

from cascade_at.core.log import get_loggers
from cascade_at.inputs.asdr import ASDR
from cascade_at.inputs.csmr import CSMR
from cascade_at.inputs.covariates import Covariate
from cascade_at.inputs.data import CrosswalkVersion
from cascade_at.inputs.demographics import Demographics
from cascade_at.inputs.locations import LocationDAG
from cascade_at.inputs.utilities.ids import get_location_set_version_id

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
        self.covariates = None

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
        self.covariates = [Covariate(
            covariate_id=c,
            demographics=self.demographics,
            decomp_step=self.decomp_step,
            gbd_round_id=self.gbd_round_id
        ).get_raw() for c in self.country_covariate_id]
        self.location_dag = LocationDAG(
            location_set_version_id=self.location_set_version_id
        )

    def modify_inputs_for_settings(self, settings):
        """

        :param settings: (cascade.settings.configuration.Configuration)
        :return:
        """
        self.asdr_for_dismod = self.asdr.configure_for_dismod()
        self.csmr_for_dismod = self.csmr.configure_for_dismod()
        self.data_for_dismod = self.data.configure_for_dismod()
        self.covariates = [c.configure_for_dismod()
                           for c in self.covariates]
        return self
