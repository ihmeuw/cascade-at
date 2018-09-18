from cascade.input_data.db import module_proxy

AGE_GROUP_SET_ID = 12

GBD_ROUND_ID = 5

METRIC_IDS = {"per_capita_rate": 3}

MEASURE_IDS = {"deaths": 1}

db_queries = module_proxy.ModuleProxy("db_queries")
db_tools = module_proxy.ModuleProxy("db_tools")
save_results = module_proxy.ModuleProxy("save_results")


def disable_databases():
    module_proxy.BLOCK_SHARED_FUNCTION_ACCESS = True
