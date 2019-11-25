import os
import dill
from pathlib import Path

from cascade_at.context.configuration import application_config
from cascade_at.core.log import get_loggers

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

        self.alchemy_file = self.inputs_dir / 'alchemy.p'
        self.inputs_file = self.inputs_dir / 'inputs.p'
        self.settings_file = self.inputs_dir / 'settings.p'

        if make:
            os.makedirs(self.inputs_dir, exist_ok=True)
            os.makedirs(self.outputs_dir, exist_ok=True)
            os.makedirs(self.database_dir, exist_ok=True)

    def db_file(self, location_id, sex_id, make=True):
        """
        Makes the database folder for a given location and sex.
        """
        folder = self.database_dir / str(location_id) / str(sex_id)
        if make:
            os.makedirs(folder, exist_ok=True)
        return folder / 'dismod.db'

    def write_inputs(self, inputs=None, alchemy=None, settings=None):
        """
        Write the inputs objects to disk.
        """
        if inputs:
            with open(self.inputs_file, "wb") as f:
                LOG.info(f"Writing input obj to {self.inputs_file}.")
                dill.dump(inputs, f)
        if alchemy:
            with open(self.alchemy_file, "wb") as f:
                LOG.info(f"Writing alchemy obj to {self.alchemy_file}.")
                dill.dump(alchemy, f)
        if settings:
            with open(self.settings_file, "wb") as f:
                LOG.info(f"Writing settings obj to {self.settings_file}.")
                dill.dump(settings, f)

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
        with open(self.alchemy_file, "rb") as f:
            LOG.info(f"Reading alchemy obj from {self.alchemy_file}.")
            alchemy = dill.load(f)
        with open(self.settings_file, "rb") as f:
            LOG.info(f"Reading settings obj from {self.settings_file}.")
            settings = dill.load(f)
        return inputs, alchemy, settings
