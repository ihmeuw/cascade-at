"""
Sequences of dismod_at commands that work together to create a cascade operation
that can be performed on a single DisMod-AT database.
"""
import inspect
from typing import List, Optional, Dict, Union

from cascade_at.jobmon.resources import DEFAULT_EXECUTOR_PARAMETERS
from cascade_at.context.arg_utils import encode_commands, encode_options, list2string


def _retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.

    Parameters
    ----------
    var
        Some variable to retrieve name from.
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class _CascadeOperation:
    def __init__(self, upstream_commands: Optional[List[str]] = None):
        if upstream_commands is None:
            upstream_commands = list()

        self.executor_parameters = DEFAULT_EXECUTOR_PARAMETERS
        self.upstream_commands = upstream_commands
        self.j_resource = False

        self.command = None

    @staticmethod
    def _script():
        raise NotImplementedError

    @staticmethod
    def _arg_to_command(k: str, v: Optional[Union[str, int, float]] = None):
        """
        Takes a key (k) and a value (v) and turns it into a command-line
        argument like k=model_version v=1 and returns --model-version 1.
        """
        k = _retrieve_name(k)
        k = '-'.join(k.split('_'))
        args = f'--{k}'
        if v is not None:
            args += f' {v}'
        return args

    def _args_to_command(self, **kwargs):
        commands = []
        for k, v in kwargs.items():
            if type(v) == bool:
                if v:
                    command = self._arg_to_command(k=k)
                else:
                    continue
            elif type(v) == list:
                command = self._arg_to_command(k=k, v=list2string(v))
            else:
                command = self._arg_to_command(k=k, v=v)
            commands.append(command)
        return ' '.join(commands)

    def _command(self, **kwargs):
        return self._script() + ' ' + self._args_to_command(**kwargs)


class ConfigureInputs(_CascadeOperation):
    def __init__(self, model_version_id: int, **kwargs):
        super().__init__(**kwargs)
        self.j_resource = True

        self.command = self._command(
            model_version_id=model_version_id,
            make=True,
            configure=True
        )

    @staticmethod
    def _script():
        return 'configure_inputs'


class _DismodDB(_CascadeOperation):
    def __init__(self, model_version_id: int,
                 parent_location_id: int, sex_id: int, fill: bool,
                 prior_parent: Optional[int] = None, prior_sex: Optional[int] = None,
                 options: Optional[Dict[str, Union[int, str, float]]] = None,
                 dm_commands: Optional[List[str]] = None, **kwargs):

        super().__init__(**kwargs)

        options = encode_options(options)
        dm_commands = encode_commands(dm_commands)

        self.command = self._command(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            fill=fill,
            prior_parent=prior_parent,
            prior_sex=prior_sex,
            options=options,
            dm_commands=dm_commands
        )

    @staticmethod
    def _script():
        return 'dismod_db'


class FitFixed(_DismodDB):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 predict: bool = True, fill: bool = True, **kwargs):
        dm_commands = ['init', 'fit fixed']
        if predict:
            dm_commands.append('predict fit_var')
        super().__init__(
            model_version_id=model_version_id, parent_location_id=parent_location_id,
            sex_id=sex_id, dm_commands=dm_commands, fill=fill, **kwargs
        )


class FitBoth(_DismodDB):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 predict: bool = True, fill: bool = True, **kwargs):
        dm_commands = [
            'init', 'fit fixed', 'set start_var fit_var', 'set scale_var fit_var', 'fit both'
        ]
        if predict:
            dm_commands.append('predict fit_var')
        super().__init__(
            model_version_id=model_version_id, parent_location_id=parent_location_id,
            sex_id=sex_id, dm_commands=dm_commands, fill=fill, **kwargs
        )


class SampleSimulate(_CascadeOperation):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 n_sim: int, n_pool: int, fit_type: str, **kwargs):
        super().__init__(**kwargs)

        self.command = self._command(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            n_sim=n_sim,
            n_pool=n_pool,
            fit_type=fit_type
        )

    @staticmethod
    def _script():
        return 'sample_simulate'


class PredictSample(_CascadeOperation):
    def __init__(self, model_version_id: int, source_location: int, source_sex: int,
                 target_locations: List[int], target_sexes: List[int], **kwargs):
        super().__init__(**kwargs)

        self.command = self._command(
            model_version_id=model_version_id,
            source_location=source_location,
            source_sex=source_sex,
            target_locations=target_locations,
            target_sexes=target_sexes,
        )

    @staticmethod
    def _script():
        return 'predict_sample'


class MulcovStatistics(_CascadeOperation):
    def __init__(self, model_version_id: int, locations: List[int], sexes: List[int],
                 outfile_name: str, sample: bool,
                 mean: bool, std: bool, quantile: Optional[List[float]], **kwargs):
        super().__init__(**kwargs)

        self.command = self._command(
            model_version_id=model_version_id,
            locations=locations,
            sexes=sexes,
            outfile_name=outfile_name,
            sample=sample,
            mean=mean,
            std=std,
            quantile=quantile
        )

    @staticmethod
    def _script():
        return 'mulcov_statistics'


class FormatAndUpload(_CascadeOperation):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int, **kwargs):
        super().__init__(**kwargs)

        self.command = self._command(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id
        )

    @staticmethod
    def _script():
        return 'format_upload'


class CleanUp(_CascadeOperation):
    def __init__(self, model_version_id: int, **kwargs):
        super().__init__(**kwargs)

        self.command = self._command(
            model_version_id=model_version_id
        )

    @staticmethod
    def _script():
        return 'cleanup'


CASCADE_OPERATIONS = {
    cls._script(): cls for cls in [
        ConfigureInputs, FitBoth, FitFixed, SampleSimulate, MulcovStatistics,
        PredictSample, FormatAndUpload, CleanUp
    ]
}
