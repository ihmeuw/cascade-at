"""
Sequences of dismod_at commands that work together to create a cascade operation
that can be performed on a single DisMod-AT database.
"""


class CascadeOperation:
    def __init__(self, model_version_id):
        self.model_version_id = model_version_id


class ConfigureInputs(CascadeOperation):
    def __init__(self, model_version_id, conn_def):
        super().__init__(model_version_id=model_version_id)
        self.model_version_id = model_version_id
        self.conn_def = conn_def

        self.command = (
            f'configure_inputs '
            f'-model-version-id {self.model_version_id} '
            f'-conn-def {self.conn_def} '
            f'--make --configure'
        )


class FitBoth(CascadeOperation):
    def __init__(self, model_version_id, parent_location_id, sex_id):
        super().__init__(model_version_id=model_version_id)
        self.parent_location_id = parent_location_id
        self.sex_id = sex_id

        self.command = (
            f'dismod_db '
            f'-model-version-id {self.model_version_id} '
            f'-parent-location-id {self.parent_location_id} '
            f'-sex-id {self.sex_id} '
            f'--commands init fit-fixed fit-random'
        )


class CleanUp(CascadeOperation):
    def __init__(self, model_version_id):
        super().__init__(model_version_id=model_version_id)

        self.command = (
            f'cleanup '
            f'-model-version-id {self.model_version_id}'
        )


CASCADE_OPERATIONS = {
    'configure_inputs': ConfigureInputs,
    'fit_both': FitBoth,
    'cleanup': CleanUp
}
