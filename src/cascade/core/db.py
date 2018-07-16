from contextlib import contextmanager

try:
    from db_tools import ezfuncs
except ImportError:

    class DummyEZFuncs:
        def __getattr__(self, name):
            raise ImportError(f"Required package db_tools not found")

    ezfuncs = DummyEZFuncs()


@contextmanager
def cursor(execution_context):
    """A context manager which exposes a database cursor connected to the database specified by
    the execution_context. The cursor will be closed when the manager exits and if it exits without
    raising an exception the connection will also be commited.
    """

    database = execution_context.parameters.database

    connection = ezfuncs.get_connection(database)
    cursor = connection.cursor()

    try:
        yield cursor
    except Exception:
        cursor.close()
        raise
    else:
        cursor.close()
        connection.commit()


def model_version_exists(execution_context):
    model_version_id = execution_context.parameters.model_version_id

    with cursor(execution_context) as c:
        query = f"""
        select exists(
                 select * from epi.model_version
                 where model_version_id = {model_version_id}
        )
        """
        c.execute(query)
        exists = c.fetchone()[0]

        return exists == 1
