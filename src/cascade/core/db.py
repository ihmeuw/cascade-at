from contextlib import contextmanager

try:
    from db_tools import ezfuncs
except ImportError:

    class DummyEZFuncs:
        def __getattr__(self, name):
            raise ImportError(f"Required package db_tools not found")

    ezfuncs = DummyEZFuncs()


@contextmanager
def cursor(execution_context=None, database=None):
    """A context manager which exposes a database cursor connected to the database specified by
    the execution_context. The cursor will be closed when the manager exits and if it exits without
    raising an exception the connection will also be commited.
    """

    with connection(execution_context, database) as c:
        cursor = c.cursor()

        try:
            yield cursor
        except Exception:
            cursor.close()
            raise
        else:
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
    try:
        yield connection
    except Exception:
        raise
    else:
        connection.commit()


def model_version_exists(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    query = f"""
    select exists(
             select * from epi.model_version
             where model_version_id = %(model_version_id)s
    )
    """

    with cursor(execution_context) as c:
        c.execute(query, args={"model_version_id": model_version_id})
        exists = c.fetchone()[0]

        return exists == 1
