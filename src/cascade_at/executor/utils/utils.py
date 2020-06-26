from cascade_at.core.db import ezfuncs, db_tools

MODEL_STATUS = {
    'Complete': 2,
    'Failed': 7,
    'Submitted': 0
}


def update_model_status(model_version_id: int, status_id: int, conn_def: str):
    """
    Updates a model status in the database.
    """
    call = (
        f"""UPDATE epi.model_version SET model_version_status_id = {status_id} 
        WHERE model_version_id = {model_version_id}"""
    )
    session = ezfuncs.get_session(conn_def)
    return db_tools.query_tools.exec_query(call, session=session, close=True)
