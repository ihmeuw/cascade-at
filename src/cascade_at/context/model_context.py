import os
from pathlib import Path
from cascade_at.context.configuration import application_config


class Context:
    def __init__(self, model_version_id, conn_def, make=False, configure_application=True):
        
        if configure_application:
            self.app = application_config()
            self.root_directory = self.app["DataLayout"]["root-directory"]
            self.cascade_dir = self.app["DataLayout"]["cascade-dir"]
        else:
            self.root_directory = Path('.')
            self.cascade_dir = 'cascade_dir'

        self.model_version_id = model_version_id
        self.conn_def = conn_def
    
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

    def db_folder(self, location_id, sex_id, make=False):
        """
        Makes the database folder for a given location and sex.
        """
        folder = self.database_dir / str(location_id) / str(sex_id)
        if make:
            os.makedirs(folder, exist_ok=True)
        return folder
    