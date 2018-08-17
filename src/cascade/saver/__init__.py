import pandas as pd

DRAWS_INPUT_FILE_PATTERN = "all_draws.h5"

INTEGRAND_ID_TO_MEASURE_ID_DF = pd.DataFrame([
    [0, 41],
    [1, 7],
    [2, 9],
    [3, 16],
    [4, 13],
    [5, 39],
    [6, 40],
    [7, 5],
    [8, 6],
    [9, 15],
    [10, 14],
    [11, 12],
    [12, 11]], columns=["integrand_id", "measure_id"])

MODEL_TITLE = "Cascade Model"
