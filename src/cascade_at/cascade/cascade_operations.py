"""
Sequences of dismod_at commands that work together to create a cascade operation
that can be performed on a single DisMod-AT database.
"""
from cascade_at.jobmon.resources import DEFAULT_EXECUTOR_PARAMETERS


class CascadeOperation:
    def __init__(self, model_version_id, upstream_commands=None):
        if upstream_commands is None:
            upstream_commands = list()

        self.model_version_id = model_version_id
        self.executor_parameters = DEFAULT_EXECUTOR_PARAMETERS
        self.upstream_commands = upstream_commands
        self.j_resource = False


class ConfigureInputs(CascadeOperation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.j_resource = True

        self.command = (
            f'configure_inputs '
            f'-model-version-id {self.model_version_id} '
            f'--make --configure'
        )


class FitBoth(CascadeOperation):
    def __init__(self, parent_location_id, sex_id, **kwargs):
        super().__init__(**kwargs)
        self.parent_location_id = parent_location_id
        self.sex_id = sex_id

        self.command = (
            f'dismod_db '
            f'-model-version-id {self.model_version_id} '
            f'-parent-location-id {self.parent_location_id} '
            f'-sex-id {self.sex_id} '
            f'--commands init fit-fixed set-start_var-fit_var set-scale_var-fit_var fit-both predict-fit_var '
        )


class SampleSimulate(CascadeOperation):
    def __init__(self, parent_location_id, sex_id, n_simulations, n_pools, fit_type, **kwargs):
        super().__init__(**kwargs)
        self.parent_location_id = parent_location_id
        self.sex_id = sex_id
        self.n_simulations = n_simulations
        self.n_pools = n_pools
        self.fit_type = fit_type

        self.command = (
            f'sample_simulate '
            f'-model-version-id {self.model_version_id} '
            f'-parent-location-id {self.parent_location_id} '
            f'-sex-id {self.sex_id} '
            f'-n-sim {self.n_simulations} '
            f'-n-pool {self.n_pools} '
            f'-fit-type {self.fit_type}'
        )


class FormatAndUpload(CascadeOperation):
    def __init__(self, parent_location_id, sex_id, **kwargs):
        super().__init__(**kwargs)
        self.parent_location_id = parent_location_id
        self.sex_id = sex_id

        self.command = (
            f'format_upload '
            f'-model-version-id {self.model_version_id} '
            f'-parent-location-id {self.parent_location_id} '
            f'-sex-id {self.sex_id}'
        )


class CleanUp(CascadeOperation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.command = (
            f'cleanup '
            f'-model-version-id {self.model_version_id}'
        )


CASCADE_OPERATIONS = {
    'configure_inputs': ConfigureInputs,
    'fit_both': FitBoth,
    'format_upload': FormatAndUpload,
    'cleanup': CleanUp
}
