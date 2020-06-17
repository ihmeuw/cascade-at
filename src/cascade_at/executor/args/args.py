from cascade_at.core import CascadeATError


class CascadeArgError(CascadeATError):
    pass


class StaticArgError(CascadeArgError):
    pass


class _Argument:
    def __init__(self, arg=None):

        self._arg = None
        self._kwargs = dict()
        if arg is not None:
            self._arg = arg

    def _to_dict(self):
        return {self._arg: self._kwargs}


class IntArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._kwargs.update({
            'type': int,
            'required': False,
            'default': None
        })
        self._kwargs.update(kwargs)


class FloatArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._kwargs.update({
            'type': float,
            'required': False,
            'default': None
        })
        self._kwargs.update(kwargs)


class StrArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._kwargs.update({
            'type': str,
            'required': False,
            'default': None
        })
        self._kwargs.update(kwargs)


class BoolArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._kwargs.update({
            'action': 'store_true',
            'required': False
        })
        self._kwargs.update(kwargs)


class ListArg(_Argument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self._kwargs.update({
            'nargs': '+',
            'required': False,
            'default': []
        })
        self._kwargs.update(kwargs)


class ModelVersionID(IntArg):
    def __init__(self):
        super().__init__()

        self._arg = '--model-version-id'
        self._kwargs.update({
            'required': True,
            'help': 'model version ID (need this from database entry)'
        })


class ParentLocationID(IntArg):
    def __init__(self):
        super().__init__()

        self._arg = '--parent-location-id'
        self._kwargs.update({
            'required': True,
            'help': 'parent location ID that determines where the database is stored'
        })


class SexID(IntArg):
    def __init__(self):
        super().__init__()

        self._arg = '--sex-id'
        self._kwargs.update({
            'required': True,
            'help': 'sex ID that determines where the database is stored'
        })


class DmCommands(ListArg):
    def __init__(self):
        super().__init__()

        self._arg = '--dm-commands'
        self._kwargs.update({
            'help': 'commands to pass to the DisMod database'
        })


class DmOptions(ListArg):
    def __init__(self):
        super().__init__()

        self._arg = '--dm-options'
        self._kwargs.update({
            'metavar': 'KEY=VALUE=TYPE',
            'help': 'options to fill in the dismod database',
            'default': None
        })


class LogLevel(StrArg):
    def __init__(self):
        super().__init__()

        self._arg = '--log-level'
        self._kwargs.update({
            'default': 'info'
        })
