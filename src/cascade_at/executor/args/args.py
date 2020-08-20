from cascade_at.core import CascadeATError


class CascadeArgError(CascadeATError):
    pass


class StaticArgError(CascadeArgError):
    pass


class _Argument:
    """
    Base class for all arguments. By default, all arguments
    are considered node_args, which is something that makes
    the command that it is used in unique within a template
    for the workflow, but not across workflows.

    This is overwritten
    if an _Argument subclass is used as a task_arg.
    """
    def __init__(self, arg=None):

        self._flag = None
        self._parser_kwargs = dict()
        if arg is not None:
            self._flag = arg
        self._task_arg = False

    def _to_dict(self):
        return {self._flag: self._parser_kwargs}


class IntArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'type': int,
            'required': False,
            'default': None
        })
        self._parser_kwargs.update(kwargs)


class FloatArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'type': float,
            'required': False,
            'default': None
        })
        self._parser_kwargs.update(kwargs)


class StrArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'type': str,
            'required': False,
            'default': None
        })
        self._parser_kwargs.update(kwargs)


class BoolArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'action': 'store_true',
            'required': False
        })
        self._parser_kwargs.update(kwargs)


class ListArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'nargs': '+',
            'required': False,
            'default': []
        })
        self._parser_kwargs.update(kwargs)


class ModelVersionID(IntArg):
    """
    The Model Version ID argument is the *only* task argument, meaning
    an argument that makes the commands that it is used in unique
    across workflows.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--model-version-id'
        self._parser_kwargs.update({
            'required': True,
            'help': 'model version ID (need this from database entry)'
        })
        self._task_arg = True


class ParentLocationID(IntArg):
    def __init__(self):
        super().__init__()

        self._flag = '--parent-location-id'
        self._parser_kwargs.update({
            'required': True,
            'help': 'parent location ID that determines where the database is stored'
        })


class SexID(IntArg):
    def __init__(self):
        super().__init__()

        self._flag = '--sex-id'
        self._parser_kwargs.update({
            'required': True,
            'help': 'sex ID that determines where the database is stored'
        })


class DmCommands(ListArg):
    def __init__(self):
        super().__init__()

        self._flag = '--dm-commands'
        self._parser_kwargs.update({
            'help': 'commands to pass to the DisMod database'
        })


class DmOptions(ListArg):
    def __init__(self):
        super().__init__()

        self._flag = '--dm-options'
        self._parser_kwargs.update({
            'metavar': 'KEY=VALUE=TYPE',
            'help': 'options to fill in the dismod database',
            'default': None
        })


class NSim(IntArg):
    def __init__(self):
        super().__init__()

        self._flag = '--n-sim'
        self._parser_kwargs.update({
            'help': 'the number of simulations to create',
            'default': 1,
            'required': False
        })


class NPool(IntArg):
    def __init__(self):
        super().__init__()

        self._flag = '--n-pool'
        self._parser_kwargs.update({
            'help': 'how many multiprocessing pools to use (default to 1 = none)',
            'default': 1,
            'required': False
        })


class LogLevel(StrArg):
    def __init__(self):
        super().__init__()

        self._flag = '--log-level'
        self._parser_kwargs.update({
            'default': 'info'
        })
