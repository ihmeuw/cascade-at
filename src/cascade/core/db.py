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

from cascade.core import CascadeError
from cascade.core.log import getLoggers
from cascade.runner.application_config import application_config

CODELOG, MATHLOG = getLoggers(__name__)

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


db_queries = ModuleProxy("db_queries")
age_spans = ModuleProxy("db_queries.get_age_metadata")
db_tools = ModuleProxy("db_tools")
ezfuncs = ModuleProxy("db_tools.ezfuncs")


def use_local_odbc_ini():
    """The password vault is an odbc.ini file that's on a drive that
    isn't accessible from all nodes, so copy it to a place it
    can be found and point the connection generation to it."""
    path = application_config()["Database"]
    local_odbc = Path(path["local-odbc"])
    have_default = (Path(path["corporate-odbc"]).exists()
                    or Path(path["personal-odbc"]).expanduser().exists())
    if local_odbc.exists() and not have_default:
        CODELOG.info(f"Using odbc.ini at {local_odbc}")
        db_tools.config.DBConfig(odbc_filepath=str(local_odbc))
    else:
        CODELOG.debug(f"Using default odbc.ini")


@contextmanager
def cursor(execution_context=None, database=None):
    """A context manager which exposes a database cursor connected to the database specified by
    either the execution_context or database if that is specified. The cursor will be closed when
    the manager exits and if it exits without raising an exception the connection will also be committed.
    """

    with connection(execution_context, database) as c:
        cursor = c.cursor()

        try:
            yield cursor
        finally:
            cursor.close()


@contextmanager
def connection(execution_context=None, database=None):
    if execution_context is None:
        if database is None:
            raise ValueError("Must supply either execution_context or database")
    else:
        if database is not None:
            raise ValueError("Must not supply both execution_context and database")
        database = execution_context.parameters.database

    CODELOG.debug(f"Calling ezfuncs.get_connection({database})")
    connection = ezfuncs.get_connection(database)
    yield connection
    connection.commit()
    connection.close()


def repeat_request(query_function):
    """
    This retries the given function if the function fails with one of
    a known set of exceptions. If it's any other exception, then it re-raises
    that exception.

    Use as::

        from cascade.core.db import db_queries, query_function
        query_function(db_queries.get_demographics)(gbd_team="epi",
                       gbd_round_id=6)

    Using this function means you would rather the program retry forever
    than that it fail when a database is down.
    """
    def repeat(*args, **kwargs):
        if hasattr(query_function, "__name__"):
            name = query_function.__name__
        else:
            name = str(query_function)

        could_work_eventually = True
        while could_work_eventually:
            try:
                result = query_function(*args, **kwargs)
            except sqlalchemy.exc.OperationalError as op_err:
                if "Lost connection" in str(op_err):
                    CODELOG.warning(f"Query {name} failed with err {op_err}. Retrying.")
                else:
                    CODELOG.warning(f"Query {name} failed with err {op_err}. Not retrying.")
                    raise
            except Exception as err:
                CODELOG.warning(
                    f"Query {name} failed with err {err}. Quitting. "
                    "If this exception looks recoverable, then add it to the list "
                    "of recoverable exceptions"
                )
                raise
            else:
                return result
            # Long sleep times because we want this to work eventually and not
            # overload the database. These failures should be rare.
            sleep(randint(60, 600))

    return repeat
