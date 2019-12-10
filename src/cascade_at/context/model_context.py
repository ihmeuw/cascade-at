import os
import dill
import json
from pathlib import Path

from cascade_at.context.configuration import application_config
from cascade_at.core.log import get_loggers
from cascade_at.inputs.covariate_specs import CovariateSpecs
from cascade_at.collector.grid_alchemy import Alchemy
from cascade_at.settings.settings import load_settings
from cascade_at.core.db import db_tools

LOG = get_loggers(__name__)


class Context:
    def __init__(self, model_version_id,
                 make=False, configure_application=True):
        """
        Context for running a model. Needs a
        :param model_version_id: (int)
        :param make: whether to make the directories for the model
        :param configure_application: configure the production application.
            If False, this can be used for testing when on a local machine.
        """
        LOG.info(f"Configuring inputs for model version {model_version_id}.")
        if configure_application:
            self.app = application_config()
            self.root_directory = self.app["DataLayout"]["root-directory"]
            self.cascade_dir = self.app["DataLayout"]["cascade-dir"]
            self.odbc_file = self.app["Database"]["local-odbc"]

            self.data_connection = 'epi'
            self.model_connection = 'dismod-at-dev'

            # Configure the odbc.ini for db-tools
            db_tools.config.DBConfig(
                load_base_defs=True,
                load_odbc_defs=True,
                odbc_filepath=self.odbc_file
            )
        else:
            self.root_directory = Path('.')
            self.cascade_dir = 'cascade_dir'

        self.model_version_id = model_version_id

        self.model_dir = (
            Path(self.root_directory) 
            / self.cascade_dir 
            / 'data' 
            / str(self.model_version_id)
        )
        self.inputs_dir = self.model_dir / 'inputs'
        self.outputs_dir = self.model_dir / 'outputs'
        self.database_dir = self.model_dir / 'dbs'

        self.inputs_file = self.inputs_dir / 'inputs.p'
        self.settings_file = self.inputs_dir / 'settings.json'

        self.log_dir = (
            Path(self.root_directory)
            / self.cascade_dir 
            / 'logs'
            / str(self.model_version_id)
        )

        if make:
            os.makedirs(self.inputs_dir, exist_ok=True)
            os.makedirs(self.outputs_dir, exist_ok=True)
            os.makedirs(self.database_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

    def db_file(self, location_id, sex_id, make=True):
        """
        Makes the database folder for a given location and sex.
        """
        folder = self.database_dir / str(location_id) / str(sex_id)
        if make:
            os.makedirs(folder, exist_ok=True)
        return folder / 'dismod.db'

    def write_inputs(self, inputs=None, settings=None):
        """
        Write the inputs objects to disk.
        """
        if inputs:
            with open(self.inputs_file, "wb") as f:
                LOG.info(f"Writing input obj to {self.inputs_file}.")
                dill.dump(inputs, f)
        if settings:
            with open(self.settings_file, 'w') as f:
                LOG.info(f"Writing settings obj to {self.settings_file}.")
                json.dump(settings, f)

    def read_inputs(self):
        """
        Read the inputs from disk.
        :return: (
            cascade_at.collector.measurement_inputs.MeasurementInputs,
            cascade_at.collector.grid_alchemy.Alchemy,
            cascade_at.collector.settings_configuration.SettingsConfiguration
        )
        """
        with open(self.inputs_file, "rb") as f:
            LOG.info(f"Reading input obj from {self.inputs_file}.")
            inputs = dill.load(f)
        with open(self.settings_file) as f:
            settings_json = json.load(f)

        settings = load_settings(settings_json=settings_json)
        alchemy = Alchemy(settings=settings)

        # For some reason the pickling process makes it so that there is a 
        # key error in FormList when trying to access CovariateSpecs

        # This re-creates the covariate specs for the inputs, but ideally
        # we don't have to do this if we can figure out why pickling makes it error.
        inputs.covariate_specs = CovariateSpecs(settings.country_covariate)

        return inputs, alchemy, settings
