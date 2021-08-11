import os
import dill
import json
from pathlib import Path
from typing import Optional

from cascade_at.context import ContextError
from cascade_at.context.configuration import application_config
from cascade_at.core.log import get_loggers
from cascade_at.inputs.covariate_specs import CovariateSpecs
from cascade_at.inputs.measurement_inputs import MeasurementInputs
from cascade_at.model.grid_alchemy import Alchemy
from cascade_at.settings.settings import load_settings
from cascade_at.settings.settings_config import SettingsConfig
from cascade_at.executor.utils.utils import MODEL_STATUS, update_model_status
from cascade_at.core.db import db_tools

LOG = get_loggers(__name__)


class Context:
    def __init__(self, model_version_id: int,
                 make: bool = False, configure_application: bool = True, root_directory: str = None):
        """
        Context for running a model.

        Arguments
        ---------
        model_version_id
            The model version ID for this context.
            If you're not configuring the application, doesn't matter what this is.
        make
            Whether or not the make the directory tree for the model.
        configure_application
            Configure the production application. If False, this can be used for testing on a local machine.
        """

        self.app = None
        self.odbc_file = None
        self.data_connection = 'epi'
        self.model_connection = 'dismod-at-dev'

        LOG.info(f"Configuring inputs for model version {model_version_id}.")
        self.app = application_config()
        self.odbc_file = self.app["Database"]["local-odbc"]
        assert ('~' not in self.odbc_file), f'The odbc file path "{self.odbc_file}" cannot start with "~".'

        if configure_application:
            self.root_directory = self.app["DataLayout"]["root-directory"]
            self.cascade_dir = self.app["DataLayout"]["cascade-dir"]

            # Configure the odbc.ini for db-tools
            db_tools.config.DBConfig(
                load_base_defs=True,
                load_odbc_defs=True,
                odbc_filepath=self.odbc_file
            )
        else:
            if root_directory is None:
                raise RuntimeError("Need a root directory to set up the files from.")
            self.root_directory = Path(root_directory)
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
        self.draw_dir = self.outputs_dir / 'draws'
        self.fit_dir = self.outputs_dir / 'fits'
        self.prior_dir = self.outputs_dir / 'priors'

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
            os.makedirs(self.draw_dir, exist_ok=True)
            os.makedirs(self.fit_dir, exist_ok=True)
            os.makedirs(self.prior_dir, exist_ok=True)
            os.makedirs(self.database_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """
        Updates status in the database.
        """
        if self.odbc_file is None:
            raise ContextError()
        if self.model_connection:
            update_model_status(
                model_version_id=self.model_version_id,
                conn_def=self.model_connection,
                status_id=MODEL_STATUS[status]
            )

    def db_folder(self, location_id: int, sex_id: int):
        os.makedirs(self.database_dir / str(location_id) / str(sex_id), exist_ok=True)
        return self.database_dir / str(location_id) / str(sex_id)

    def db_file(self, location_id: int, sex_id: int) -> Path:
        """
        Gets the database file for a given location and sex.

        Parameters
        ---------
        location_id
            Location ID for the database (parent).
        sex_id
            Sex ID for the database, as the reference.
        """
        return self.db_folder(location_id, sex_id) / 'dismod.db'

    def db_index_file_pattern(self, location_id: int, sex_id: int) -> str:
        """
        Gets the database file pattern for databases with indices. Used
        in sample simulate when it's done in parallel.

        Parameters
        ----------
        location_id
            Location ID for the database (parent).
        sex_id
            Sex ID for the database, as the reference.

        Returns
        -------
        String representing the absolute path to the index database.
        """
        return str(self.db_folder(location_id, sex_id)) + '/dismod_{index}.db'

    def write_inputs(self, inputs: Optional[MeasurementInputs] = None,
                     settings: Optional[SettingsConfig] = None):
        """
        Write the inputs objects to disk.
        """
        if inputs:
            os.makedirs(self.inputs_file.parent, exist_ok = True)
            with open(self.inputs_file, "wb") as f:
                LOG.info(f"Writing input obj to {self.inputs_file}.")
                dill.dump(inputs, f)
        if settings:
            os.makedirs(self.inputs_file.parent, exist_ok = True)
            with open(self.settings_file, 'w') as f:
                LOG.info(f"Writing settings obj to {self.settings_file}.")
                json.dump(settings, f)

    def read_inputs(self) -> (MeasurementInputs, Alchemy, SettingsConfig):
        """
        Read the inputs from disk.
        """
        with open(self.inputs_file, "rb") as f:
            LOG.info(f"Reading input obj from {self.inputs_file}.")
            inputs = dill.load(f)
        with open(self.settings_file) as f:
            LOG.info(f"Reading json file from {self.settings_file}.")
            settings_json = json.load(f)
        settings = load_settings(settings_json=settings_json)
        alchemy = Alchemy(settings=settings)

        # For some reason the pickling process makes it so that there is a 
        # key error in FormList when trying to access CovariateSpecs
        # This re-creates the covariate specs for the inputs, but ideally
        # we don't have to do this if we can figure out why pickling makes it error.
        inputs.covariate_specs = CovariateSpecs(
            country_covariates=settings.country_covariate,
            study_covariates=settings.study_covariate
        )
        return inputs, alchemy, settings
