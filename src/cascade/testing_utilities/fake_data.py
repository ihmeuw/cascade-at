from types import SimpleNamespace

import numpy as np
import pandas as pd

from cascade.core import getLoggers
from cascade.core.db import db_queries
from cascade.executor.covariate_data import find_covariate_names
from cascade.input_data.db.locations import get_descendants
from cascade.input_data.db.locations import location_hierarchy
from cascade.model.integrands import make_average_integrand_cases_from_gbd

CODELOG, MATHLOG = getLoggers(__name__)


def retrieve_fake_data(execution_context, local_settings, covariate_data_spec):
    """Like :py:func:`cascade.executor.estimate_location` except makes
    all fake data."""
    data = SimpleNamespace()
    data_access = local_settings.data_access
    parent_id = local_settings.parent_location_id

    data.locations = location_hierarchy(
        data_access.gbd_round_id, location_set_version_id=data_access.location_set_version_id)
    children = get_descendants(data.locations, parent_id, children_only=True)
    seqs = list(range(10, 110))
    data.bundle = pd.DataFrame(dict(
        seq=seqs,
        measure=np.repeat(["prevalence", "Sincidence", "mtother", "mtexcess"], 25),
        mean=np.repeat([0.01, 0.02, 0.03, 0.04], 25),
        sex=np.repeat(["male", "female", "both", "female"], 25),
        sex_id=np.repeat([1, 2, 3, 2], 25),
        standard_error=np.repeat([0.005, 0.003, 0.004, 0.002], 25),
        hold_out=np.repeat([0, 0, 0, 1], 25),
        age_lower=np.repeat([0.0, 5.0, 2.0, 10], 25),
        age_upper=np.repeat([0.019, 10.0, 100.0, 80], 25),
        time_lower=np.repeat([1990, 2000, 2005, 2007], 25),
        time_upper=np.repeat([1991, 2005, 2020, 2008], 25),
        location_id=np.repeat(list(children), 100)[:len(seqs)],
    ))

    study_ids = list(set(st_set.study_covariate_id for st_set in local_settings.settings.study_covariate))

    if study_ids:
        cov_seq = list()
        sid = list()
        for idx, s in enumerate(seqs):
            if idx % 4 != 0:
                cov_seq.append(s)
                sid.append(study_ids[idx % len(study_ids)])
        data.sparse_covariate_data = pd.DataFrame(dict(
            study_covariate_id=sid,
            seq=cov_seq,
            bundle_id=data_access.bundle_id,
        ))
    else:
        data.sparse_covariate_data = pd.DataFrame(
            columns=["study_covariate_id", "seq", "bundle_id"])

    data.ages_df = db_queries.get_age_metadata(
        age_group_set_id=data_access.age_group_set_id,
        gbd_round_id=data_access.gbd_round_id
    )
    data.years_df = db_queries.get_demographics(
        gbd_team="epi", gbd_round_id=data_access.gbd_round_id)["year_id"]

    include_birth_prevalence = local_settings.settings.model.birth_prev
    data.average_integrand_cases = \
        make_average_integrand_cases_from_gbd(
            data.ages_df, data.years_df, local_settings.sex_id,
            local_settings.parent_location_id, include_birth_prevalence)

    data.study_id_to_name, data.country_id_to_name = find_covariate_names(
        execution_context, covariate_data_spec)

    # These are the draws as output of the parent location.
    data.draws = None

    # The parent can also supply integrands as a kind of prior.
    # These will be shaped like input measurement data.
    data.integrands = None
    return data
