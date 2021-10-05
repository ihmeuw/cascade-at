import db_tools
def execute_select(query, conn_def = 'dismod-at-dev'):
    df = db_tools.ezfuncs.query(query, conn_def = conn_def)
    return df

