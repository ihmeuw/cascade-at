"""This module prvides basic database access tools.

All other code which accesses the external databases should do so through the context managers defined here so we
have consistency and a single chokepoint for that access.
"""
import importlib
from contextlib import contextmanager

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)

BLOCK_SHARED_FUNCTION_ACCESS = False
"""
Used to control access to the testing environment. You can't load this
with from <module> import BLOCK_SHARED_FUNCTION_ACCESS. You have to
modify the value as ``module_proxy.BLOCK_SHARED_FUNCTION_ACCESS``.
"""


class DatabaseSandboxViolation(Exception):
    """Attempted to call a module that is intentionally restricted in the current environment."""


class ModuleProxy:
    """
    This class acts like a module. It's meant to be imported into an init.
    This exists in order to actively turn off modules during testing.
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
save_results = ModuleProxy("save_results")
dbtrees = ModuleProxy("hierarchies.dbtrees")


@contextmanager
def cursor(execution_context=None, database=None):
    """A context manager which exposes a database cursor connected to the database specified by
    either the execution_context or database if that is specifed. The cursor will be closed when
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

    connection = ezfuncs.get_connection(database)
    yield connection
    connection.commit()


def model_version_exists(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = """
    select exists(
             select * from epi.model_version
             where model_version_id = %(model_version_id)s
    )
    """

    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        exists = c.fetchone()[0]

        return exists == 1


def latest_model_version(execution_context):
    model_id = execution_context.parameters.modelable_entity_id

    query = """
    select model_version_id from epi.model_version
    where modelable_entity_id = %(modelable_entity_id)s
    order by last_updated desc
    limit 1
    """

    with cursor(execution_context) as c:
        c.execute(query, args={"modelable_entity_id": model_id})
        result = c.fetchone()
        if result is not None:
            return result[0]
        else:
            raise RuntimeError(
                f"No model version for modelable entity id {model_id} in database.")
