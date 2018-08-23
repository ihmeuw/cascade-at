try:
    import db_queries
except ImportError:

    class DummyDBQueries:
        def __getattr__(self, name):
            raise ImportError(f"Required package db_queries not found")

    db_queries = DummyDBQueries()

AGE_GROUP_SET_ID = 12

GBD_ROUND_ID = 5

METRIC_IDS = {"per_capita_rate": 3}

MEASURE_IDS = {"deaths": 1}
