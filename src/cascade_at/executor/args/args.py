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
    """
    An integer argument.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'type': int,
            'required': False,
            'default': None
        })
        self._parser_kwargs.update(kwargs)


class FloatArg(_Argument):
    """
    A float argument.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'type': float,
            'required': False,
            'default': None
        })
        self._parser_kwargs.update(kwargs)


class StrArg(_Argument):
    """
    A string argument.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'type': str,
            'required': False,
            'default': None
        })
        self._parser_kwargs.update(kwargs)


class BoolArg(_Argument):
    """
    A boolean argument.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._parser_kwargs.update({
            'action': 'store_true',
            'required': False
        })
        self._parser_kwargs.update(kwargs)


class ListArg(_Argument):
    """
    A list argument. Passed in as an ``nargs +`` type of argument
    to ``argparse``.
    """
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
    """
    A parent location ID argument.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--parent-location-id'
        self._parser_kwargs.update({
            'required': True,
            'help': 'parent location ID that determines where the database is stored'
        })


class SexID(IntArg):
    """
    A sex ID argument.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--sex-id'
        self._parser_kwargs.update({
            'required': True,
            'help': 'sex ID that determines where the database is stored'
        })


class DmCommands(ListArg):
    """
    A dismod commands argument, based off of the list
    argument.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--dm-commands'
        self._parser_kwargs.update({
            'help': 'commands to pass to the DisMod database'
        })


class DmOptions(ListArg):
    """
    A dismod options argument, based off of the list
    argument. Arguments need to be passed in as a list,
    but then look like ``KEY=VALUE=TYPE``. So, if you wanted
    the options to look like this ``{'kind': 'random'}``,
    you would pass on the command-line ``kind=random=str``.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--dm-options'
        self._parser_kwargs.update({
            'metavar': 'KEY=VALUE=TYPE',
            'help': 'options to fill in the dismod database',
            'default': None
        })


class NSim(IntArg):
    """
    Number of simulations argument. Defaults to 1.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--n-sim'
        self._parser_kwargs.update({
            'help': 'the number of simulations to create',
            'default': 1,
            'required': False
        })


class NPool(IntArg):
    """
    Number of threads for a multiprocessing pool argument, defaults to 1, which
    is no multiprocessing.
    """
    def __init__(self):
        super().__init__()

        self._flag = '--n-pool'
        self._parser_kwargs.update({
            'help': 'how many multiprocessing pools to use (default to 1 = none)',
            'default': 1,
            'required': False
        })


class LogLevel(StrArg):
    """
    Logging level argument. Defaults to "info".
    """
    def __init__(self):
        super().__init__()

        self._flag = '--log-level'
        self._parser_kwargs.update({
            'default': 'info'
        })
