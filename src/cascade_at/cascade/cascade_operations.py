"""
Sequences of dismod_at commands that work together to create a cascade operation
that can be performed on a single DisMod-AT database.
"""
from typing import List, Optional, Dict, Union

from cascade_at.jobmon.resources import DEFAULT_EXECUTOR_PARAMETERS
from cascade_at.executor.args.arg_utils import encode_commands, encode_options, list2string
from cascade_at.executor.args.executor_args import ARG_DICT
from cascade_at.core import CascadeATError


class CascadeOperationValidationError(CascadeATError):
    pass


def _arg_to_flag(name: str) -> str:
    arg = '-'.join(name.split('_'))
    return f'--{arg}'


def _arg_to_command(k: str, v: Optional[Union[str, int, float]] = None):
    """
    Takes a key (k) and a value (v) and turns it into a command-line
    argument like k=model_version v=1 and returns --model-version 1.
    """
    command = _arg_to_flag(k)
    if v is not None:
        command += f' {v}'
    return command


def _args_to_command(**kwargs):
    commands = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if type(v) == bool:
            if v:
                command = _arg_to_command(k=k)
            else:
                continue
        elif type(v) == list:
            command = _arg_to_command(k=k, v=list2string(v))
        else:
            command = _arg_to_command(k=k, v=v)
        commands.append(command)
    return ' '.join(commands)


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

    def _make_command(self, **kwargs):
        return self._script() + ' ' + _args_to_command(**kwargs)

    def _validate(self, **kwargs):
        if self._script() not in ARG_DICT:
            raise CascadeOperationValidationError(f"Cannot find script args for {self._script()}. "
                                                  f"Valid scripts are {ARG_DICT.keys()}.")
        arg_list = ARG_DICT[self._script()]
        kwargs = {
            _arg_to_flag(k): v for k, v in kwargs.items()
        }
        for k, v in arg_list.argument_dict.items():
            if v['required']:
                if k not in kwargs:
                    raise CascadeATError(f"Missing argument {k} for script {self._script()}.")
                if 'type' in v:
                    assert type(kwargs[k]) == v['type']
        for k, v in kwargs.items():
            if k not in arg_list.argument_dict:
                raise CascadeATError(f"Tried to pass argument {k} but that is not in the allowed list"
                                     f"of arguments for {self._script()}: {list(arg_list.argument_dict.keys())}.")

    def _configure(self, **command_args):
        self._validate(**command_args)
        self.command = self._make_command(**command_args)


class ConfigureInputs(_CascadeOperation):
    def __init__(self, model_version_id: int, **kwargs):
        super().__init__(**kwargs)
        self.j_resource = True

        self._configure(
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
                 dm_options: Optional[Dict[str, Union[int, str, float]]] = None,
                 dm_commands: Optional[List[str]] = None,
                 save_prior: bool = False,
                 save_fit: bool = False,
                 **kwargs):

        super().__init__(**kwargs)

        if dm_options is not None:
            dm_options = encode_options(dm_options)
        if dm_commands is not None:
            dm_commands = encode_commands(dm_commands)

        self._configure(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            fill=fill,
            prior_parent=prior_parent,
            prior_sex=prior_sex,
            dm_options=dm_options,
            dm_commands=dm_commands,
            save_prior=save_prior,
            save_fit=save_fit
        )

    @staticmethod
    def _script():
        return 'dismod_db'


class Fit(_DismodDB):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 predict: bool = True, fill: bool = True, both: bool = False,
                 save_fit: bool = False, save_prior: bool = False, **kwargs):

        dm_commands = ['init', 'fit fixed']
        if both:
            dm_commands += [
                'set start_var fit_var', 'set scale_var fit_var', 'fit both'
            ]
        if predict:
            dm_commands.append('predict fit_var')
        if save_fit and not predict:
            raise CascadeOperationValidationError("Can't save results if you don't predict first.")
        super().__init__(
            model_version_id=model_version_id, parent_location_id=parent_location_id,
            sex_id=sex_id, dm_commands=dm_commands, fill=fill,
            save_fit=save_fit, save_prior=save_prior, **kwargs
        )


class SampleSimulate(_CascadeOperation):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 n_sim: int, n_pool: int, fit_type: str, **kwargs):
        super().__init__(**kwargs)

        self._configure(
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


class Predict(_CascadeOperation):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 child_locations: List[int], child_sexes: List[int],
                 prior_grid: bool = True, save_fit: bool = False,
                 sample: bool = True, **kwargs):

        super().__init__(**kwargs)

        self._configure(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            child_locations=child_locations,
            child_sexes=child_sexes,
            prior_grid=prior_grid,
            save_fit=save_fit,
            sample=sample
        )

    @staticmethod
    def _script():
        return 'predict'


class MulcovStatistics(_CascadeOperation):
    def __init__(self, model_version_id: int, locations: List[int], sexes: List[int],
                 outfile_name: str, sample: bool,
                 mean: bool, std: bool, quantile: Optional[List[float]], **kwargs):
        super().__init__(**kwargs)

        self._configure(
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


class Upload(_CascadeOperation):
    def __init__(self, model_version_id: int, final: bool = False, fit: bool = False,
                 prior: bool = False, **kwargs):
        super().__init__(**kwargs)

        self._configure(
            model_version_id=model_version_id,
            final=final, fit=fit, prior=prior
        )

    @staticmethod
    def _script():
        return 'upload'


class CleanUp(_CascadeOperation):
    def __init__(self, model_version_id: int, **kwargs):
        super().__init__(**kwargs)

        self._configure(
            model_version_id=model_version_id
        )

    @staticmethod
    def _script():
        return 'cleanup'


CASCADE_OPERATIONS = {
    cls._script(): cls for cls in [
        ConfigureInputs, _DismodDB, SampleSimulate, MulcovStatistics,
        Predict, Upload, CleanUp
    ]
}
