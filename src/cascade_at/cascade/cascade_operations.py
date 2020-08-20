"""
Sequences of dismod_at commands that work together to create a cascade operation
that can be performed on a single DisMod-AT database.
"""
from typing import List, Optional, Dict, Union, Any

from cascade_at.core import CascadeATError
from cascade_at.executor.args.arg_utils import _args_to_command, _arg_to_flag, ArgumentList
from cascade_at.executor.args.arg_utils import _flag_to_arg
from cascade_at.executor.args.arg_utils import encode_commands, encode_options
from cascade_at.executor.args.args import BoolArg
from cascade_at.executor.args.executor_args import ARG_DICT
from cascade_at.jobmon.resources import DEFAULT_EXECUTOR_PARAMETERS


class CascadeOperationValidationError(CascadeATError):
    pass


class _CascadeOperation:
    def __init__(self, upstream_commands: Optional[List[str]] = None,
                 executor_parameters: Optional[Dict[str, Any]] = None):

        if upstream_commands is None:
            upstream_commands = list()
        self.upstream_commands: List[str] = upstream_commands

        self.executor_parameters = DEFAULT_EXECUTOR_PARAMETERS
        if executor_parameters is not None:
            self.executor_parameters.update(executor_parameters)

        self.j_resource: bool = False

        self.name: Optional[str] = None
        self.command: Optional[str] = None
        self.template_kwargs: Optional[Dict[str, str]] = dict()
        self.name_components: List = []
        self.arg_list: ArgumentList = ARG_DICT[self._script()]

    @staticmethod
    def _script():
        raise NotImplementedError

    def _make_template(self):
        return self._script() + ' ' + self.arg_list.template

    def _make_command(self, **kwargs):
        return self._script() + ' ' + _args_to_command(**kwargs)

    def _make_name(self):
        return 'dmat_' + self._script() + '_' + '_'.join([str(x) for x in self.name_components])

    def _validate(self, **kwargs):
        if self._script() not in ARG_DICT:
            raise CascadeOperationValidationError(f"Cannot find script args for {self._script()}. "
                                                  f"Valid scripts are {ARG_DICT.keys()}.")
        kwargs = {
            _arg_to_flag(k): v for k, v in kwargs.items()
        }
        for k, v in self.arg_list.argument_dict.items():
            if v['required']:
                if k not in kwargs:
                    raise CascadeATError(f"Missing argument {k} for script {self._script()}.")
                if 'type' in v and 'nargs' not in v:
                    if type(kwargs[k]) != v['type']:
                        raise CascadeATError(
                            f"Expected {k} arg type is {v['type']} but got {type(kwargs[k])}."
                        )
                elif 'nargs' in v:
                    if v['nargs'] == '+':
                        if not isinstance(kwargs[k], list):
                            raise CascadeATError(f"{k} should be a list.")
                        if 'type' in v:
                            for i in kwargs[k]:
                                if type(i) != v['type']:
                                    raise CascadeATError(
                                        f"Expected list arg {k} is {v['type']} but got {type(kwargs[k])}"
                                    )
                else:
                    pass

        for k, v in kwargs.items():
            if k not in self.arg_list.argument_dict:
                raise CascadeATError(f"Tried to pass argument {k} but that is not in the allowed list"
                                     f"of arguments for {self._script()}: {list(self.arg_list.argument_dict.keys())}.")

    def _make_template_kwargs(self, **kwargs) -> Dict[str, str]:
        """
        Takes kwargs like model_version_id=0
        and turns it into kwargs dict that looks
        like {'model_version_id': --model-version-id 0}.
        For boolean args, it will look like
        {'do_this': '--do-this'}. And for arguments
        from self.arg_list that have defaults, it will
        fill in the default value if it is not passed in
        the kwargs (unless it's None).

        Parameters
        ----------
        kwargs
            Keyword arguments

        Returns
        -------
        Dictionary of keyword arguments similar to what was passed but with
        values that have been converted to what the TaskTemplate expects. Also
        filling in default arguments that are not passed but are listed in
        the ArgumentList for self.
        """
        template_kwargs = dict()
        passed_args = {
            _arg_to_flag(k): v for k, v in kwargs.items()
        }

        for argument in self.arg_list.arg_list:
            # The default value for each arg
            # will be an empty string, and is only
            # overwritten if it has a passed_arg
            # or a default.
            value = ""
            arg = _flag_to_arg(argument._flag)
            if isinstance(argument, BoolArg):
                if argument._flag in passed_args:
                    if passed_args[argument._flag]:
                        value = _args_to_command(**{
                            arg: passed_args[argument._flag]
                        })
            else:
                if argument._flag in passed_args:
                    value = _args_to_command(**{
                        arg: passed_args[argument._flag]
                    })
                else:
                    if argument._parser_kwargs['default'] is not None:
                        value = _args_to_command(**{
                            arg: argument._parser_kwargs['default']
                        })
            template_kwargs.update({
                _flag_to_arg(argument._flag): value
            })
        return template_kwargs

    def _configure(self, **command_args):
        self._validate(**command_args)
        self.command = self._make_command(**command_args)
        self.name = self._make_name()
        self.template_kwargs = self._make_template_kwargs(**command_args)


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
                 prior_samples: bool = False, prior_mulcov: bool = False,
                 prior_parent: Optional[int] = None, prior_sex: Optional[int] = None,
                 dm_options: Optional[Dict[str, Union[int, str, float]]] = None,
                 dm_commands: Optional[List[str]] = None,
                 save_prior: bool = False,
                 save_fit: bool = False,
                 **kwargs):

        super().__init__(**kwargs)
        self.name_components = [model_version_id, parent_location_id, sex_id]

        if dm_options is not None:
            dm_options = encode_options(dm_options)
        if dm_commands is not None:
            dm_commands = encode_commands(dm_commands)

        self._configure(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            fill=fill,
            prior_mulcov=prior_mulcov,
            prior_samples=prior_samples,
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


class Sample(_CascadeOperation):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 n_sim: int, fit_type: str, asymptotic: bool, n_pool: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.name_components = [model_version_id, parent_location_id, sex_id]

        self._configure(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            n_sim=n_sim,
            n_pool=n_pool,
            fit_type=fit_type,
            asymptotic=asymptotic
        )

    @staticmethod
    def _script():
        return 'sample'


class Predict(_CascadeOperation):
    def __init__(self, model_version_id: int, parent_location_id: int, sex_id: int,
                 child_locations: Optional[List[int]] = None, child_sexes: Optional[List[int]] = None,
                 prior_grid: bool = True, save_fit: bool = False, save_final: bool = False,
                 sample: bool = True, **kwargs):

        super().__init__(**kwargs)
        self.name_components = [model_version_id, parent_location_id, sex_id]

        self._configure(
            model_version_id=model_version_id,
            parent_location_id=parent_location_id,
            sex_id=sex_id,
            child_locations=child_locations,
            child_sexes=child_sexes,
            prior_grid=prior_grid,
            save_fit=save_fit,
            save_final=save_final,
            sample=sample
        )

    @staticmethod
    def _script():
        return 'predict'


class MulcovStatistics(_CascadeOperation):
    def __init__(self, model_version_id: int, locations: List[int], sexes: List[int],
                 sample: bool,
                 mean: bool, std: bool, quantile: Optional[List[float]],
                 outfile_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.name_components = [model_version_id]

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
        self.name_components = [model_version_id]

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
        self.name_components = [model_version_id]

        self._configure(
            model_version_id=model_version_id
        )

    @staticmethod
    def _script():
        return 'cleanup'


CASCADE_OPERATIONS = {
    cls._script(): cls for cls in [
        ConfigureInputs, _DismodDB, Sample, MulcovStatistics,
        Predict, Upload, CleanUp
    ]
}
