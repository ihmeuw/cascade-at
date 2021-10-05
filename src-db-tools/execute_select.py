import db_tools
def execute_select(query, conn_def = 'epi'):
    df = db_tools.ezfuncs.query(query, conn_def = conn_def)
    return df

