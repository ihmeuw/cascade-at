"""This module provides basic database access tools.

All other code which accesses the external databases should do so through the context managers defined here so we
have consistency and a single choke point for that access.
"""
import importlib
from contextlib import contextmanager
from pathlib import Path
from random import randint
from time import sleep

import sqlalchemy

from cascade_at.core.errors import CascadeError
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)

BLOCK_SHARED_FUNCTION_ACCESS = False
"""
Used to control access to the testing environment. You can't load this
with from <module> import BLOCK_SHARED_FUNCTION_ACCESS. You have to
modify the value as ``module_proxy.BLOCK_SHARED_FUNCTION_ACCESS``.
"""


class DatabaseSandboxViolation(CascadeError):
    """Attempted to call a module that is intentionally restricted in the current environment."""


class ModuleProxy:
    """
    This class acts like a module. It's meant to be imported into an init.
    This exists in order to actively turn off modules during testing.
    Ensure tests that claim not to use database functions
    really don't use them, so that their tests also pass outside IHME.

    Examples
    --------
    >>> # db-queries and db-tools are IHME internal packages
    >>>
    >>> db_queries = ModuleProxy("db_queries")
    >>> ezfuncs = ModuleProxy("db_tools.ezfuncs")
    """
    def __init__(self, module_name):
        if not isinstance(module_name, str):
            raise ValueError(f"This accepts a module name, not the module itself.")

        self.name = module_name
        try:
            self._module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            self._module = None

    def __getattr__(self, name):
        if BLOCK_SHARED_FUNCTION_ACCESS:
            raise DatabaseSandboxViolation(
                f"Illegal access to module {self.name}. Are you trying to use "
                f"the shared functions in a unit test?")

        if self._module:
            return getattr(self._module, name)
        else:
            raise ModuleNotFoundError(
                f"The module {self.name} could not be imported in this environment. "
                f"Failed to call {self.name}.{name}."
            )

    def __dir__(self):
        return dir(self._module)

db_tools = ModuleProxy("db_tools")
ezfuncs = ModuleProxy("db_tools.ezfuncs")
gbd = ModuleProxy("gbd")
decomp_step = ModuleProxy("gbd.decomp_step")
elmo = ModuleProxy("elmo")

import sys
if 'darwin' in sys.platform:    # gma Something, perhaps db_tools, is importing db_queries incorrectly causing a deprecation error
    del sys.modules['db_queries']
db_queries = ModuleProxy("db_queries")

if 'darwin' in sys.platform:    # gma Add logic to skip jobmon imports if jobmon switch is not set at top level call
    LOG.warning("FIXME -- gma -- Add logic to skip jobmon imports if jobmon switch is not set at top level call")
    swarm = None
    api = None
    task = None
    task_template = None
    sge = None
else:
    swarm = ModuleProxy("jobmon.client.swarm")
    api = ModuleProxy("jobmon.client.api")
    task = ModuleProxy("jobmon.client.task")
    task_template = ModuleProxy("jobmon.client.task_template")
    sge = ModuleProxy("jobmon.client.execution.strategies.sge")
