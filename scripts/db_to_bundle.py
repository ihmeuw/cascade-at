import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.dismod.constants import DensityEnum, IntegrandEnum
from cascade.dismod.db.wrapper import get_engine
from cascade.input_data.configuration.id_map import make_integrand_map

MEASURES_ACCEPTABLE_TO_ELMO = {
    "prevalence",
    "duration",
    "yld",
    "continuous",
    "cfr",
    "proportion",
    "mtstandard",
    "relrisk",
    "incidence",
    "tincidence",
    "sincidence",
    "remission",
    "mtexcess",
    "pmtexcess",
    "mtwith",
    "mtall",
    "mtspecific",
    "mtother",
}

MEASURE_ID_TO_CANONICAL_NAME = {
    24: "acute_inc",
    23: "acute_prev",
    17: "cfr",
    22: "chronic_prev",
    19: "continuous",
    2: "daly",
    1: "death",
    21: "diswght",
    8: "duration",
    45: "fertility",
    28: "hale",
    43: "haq_index",
    6: "incidence",
    26: "le",
    37: "le_decomp",
    30: "le_nsnh",
    31: "le_nswh",
    36: "lt_prevalence",
    25: "mmr",
    34: "mort_risk",
    14: "mtall",
    9: "mtexcess",
    16: "mtother",
    15: "mtspecific",
    12: "mtstandard",
    13: "mtwith",
    38: "pini",
    10: "pmtexcess",
    27: "pod",
    32: "pod_nsnh",
    33: "pod_nswh",
    44: "population",
    5: "prevalence",
    18: "proportion",
    11: "relrisk",
    7: "remission",
    29: "sev",
    41: "sincidence",
    35: "st_prevalence",
    20: "survival_rate",
    39: "susceptible",
    42: "tincidence",
    40: "withc",
    3: "yld",
    4: "yll",
}
REQUIRED_COLUMNS = [
    "bundle_id",
    "seq",
    "nid",
    "underlying_nid",
    "input_type",
    "source_type",
    "location_id",
    "sex",
    "year_start",
    "year_end",
    "age_start",
    "age_end",
    "measure",
    "mean",
    "lower",
    "upper",
    "standard_error",
    "effective_sample_size",
    "cases",
    "sample_size",
    "unit_type",
    "unit_value_as_published",
    "uncertainty_type",
    "uncertainty_type_value",
    "representative_name",
    "urbanicity_type",
    "recall_type",
    "recall_type_value",
    "sampling_type",
    "group",
    "specificity",
    "group_review",
    "is_outlier",
    "design_effect",
]

DUMMY_VALUES = {
    "nid": 119_796,
    "source_type": "Unidentifiable",
    "unit_type": "Person",
    "unit_value_as_published": 1,
    "representative_name": "Nationally representative only",
    "urbanicity_type": "Mixed/both",
    "recall_type": "Period: years",
    "recall_type_value": 1,
    "is_outlier": 0,
    "response_rate": "",
}


def main():
    readable_by_all = 0o0002
    os.umask(readable_by_all)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("bundle_id", type=int)
    args = parser.parse_args()

    engine = get_engine(Path(args.input_file))

    data = pd.read_sql_query("select * from data", engine)
    node_table = pd.read_sql_query("select * from node", engine)
    covariate_table = pd.read_sql_query("select * from covariate", engine)

    # For distributions other than gaussian remove the standard deviation
    # which will cause Elmo to use Wilson Score Interval to estimate
    # uncertainty from sample size and Theo says that's what we want
    data.loc[data.density_id != DensityEnum.gaussian.value, "meas_std"] = np.nan

    covariate_columns = [c for c in data.columns if c.startswith("x_")]
    data = data[
        [
            "age_lower",
            "age_upper",
            "hold_out",
            "node_id",
            "time_lower",
            "time_upper",
            "sample_size",
            "integrand_id",
            "meas_value",
            "meas_std",
        ]
        + covariate_columns
    ]
    data = data.rename(
        columns={
            "age_lower": "age_start",
            "age_upper": "age_end",
            "time_lower": "year_start",
            "time_upper": "year_end",
            "meas_value": "mean",
            "meas_std": "standard_error",
        }
    )

    # Covariates
    cov_pattern = re.compile("[sc]_(.*)_[^_]+")
    dm_cov_to_gbd_study_cov = {
        f"x_{r['covariate_id']}": f"cv_{cov_pattern.match(r['covariate_name']).group(1)}"
        for _, r in covariate_table.iterrows()
        if r["covariate_name"].startswith("s_")
    }
    for c in covariate_columns:
        if c in dm_cov_to_gbd_study_cov:
            data = data.rename(columns={c: dm_cov_to_gbd_study_cov[c]})
        else:
            data = data.drop(c, axis=1)

    # Convert sex covariate to sex name
    data["sex"] = data.cv_sex.apply(lambda c: {-0.5: "Female", 0.5: "Male", 0.0: "Both"}[c])
    data = data.drop("cv_sex", axis=1)

    # Convert nodes to locations
    node_to_location = {r.node_id: r.c_location_id for _, r in node_table.iterrows()}
    data["location_id"] = data.node_id.apply(lambda nid: node_to_location[nid])
    data = data.drop("node_id", axis=1)

    # Convert integrands to measures
    integrand_to_measure = {v.value: MEASURE_ID_TO_CANONICAL_NAME[k] for k, v in make_integrand_map().items()}
    # prevalence and incidence are special because they have more complicated relationships with integrands
    # than other measures so clean them up
    integrand_to_measure[IntegrandEnum.prevalence.value] = "prevalence"
    integrand_to_measure[IntegrandEnum.Tincidence.value] = "tincidence"
    integrand_to_measure[IntegrandEnum.Sincidence.value] = "sincidence"

    data["measure"] = data.integrand_id.apply(integrand_to_measure.get)
    data = data.drop("integrand_id", axis=1)

    assert not set(data.measure.unique()) - MEASURES_ACCEPTABLE_TO_ELMO

    # Add in the bundle_id
    data = data.assign(bundle_id=args.bundle_id)

    data = data.assign(**DUMMY_VALUES)
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    data = data.assign(**{c: "" for c in missing_columns})

    data.to_excel(args.output_file, "extraction", index=False)


if __name__ == "__main__":
    main()
