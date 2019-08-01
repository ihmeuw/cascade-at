from math import nan
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.random import RandomState

from cascade.core import getLoggers
from cascade.core.db import db_queries
from cascade.executor.covariate_data import assign_epiviz_covariate_names
from cascade.executor.covariate_data import find_covariate_names
from cascade.input_data.configuration.raw_input import validate_input_data_types
from cascade.input_data.db.locations import (
    get_descendants, all_locations_with_these_parents, location_hierarchy
)
from cascade.model.integrands import make_average_integrand_cases_from_gbd

CODELOG, MATHLOG = getLoggers(__name__)


def fake_asdr(locations_for_asdr, sex, ages):
    age_list = [
        (x, y) for (i, x, y) in
        ages[["age_group_years_start", "age_group_years_end"]].to_records()
    ]
    df = pd.DataFrame(
        [dict(
            location=loc_id,
            sex_id=sex_id,
            hold_out=0,
            age_lower=when[0],
            age_upper=when[1],
            time_lower=year,
            time_upper=year + 1,
            mean=0.01,
            std=0.002,
            eta=nan,
            nu=nan,
        )
            for loc_id in locations_for_asdr
            for sex_id in sex
            for year in range(1950, 2020)
            for when in age_list
        ]
    )
    return df


def fake_csmr(locations_for_asdr, sex, ages):
    age_list = [
        (x, y) for (i, x, y) in
        ages[["age_group_years_start", "age_group_years_end"]].to_records()
    ]
    df = pd.DataFrame(
        [dict(
            location_id=loc_id,
            sex_id=sex_id,
            age_lower=when[0],
            age_upper=when[1],
            time_lower=year,
            time_upper=year + 1,
            mean=0.01,
            lower=0.009,
            upper=0.011,
        )
            for loc_id in locations_for_asdr
            for sex_id in sex
            for year in range(1950, 2020)
            for when in age_list
        ]
    )
    return df


def retrieve_fake_data(execution_context, local_settings, covariate_data_spec, rng=None):
    """Like :py:func:`cascade.executor.estimate_location` except makes
    all fake data. This is deterministic."""
    if rng is None:
        rng = RandomState(298472943)
    elif isinstance(rng, int):
        rng = RandomState(rng)
    else:
        assert isinstance(rng, RandomState)

    data = SimpleNamespace()
    data_access = local_settings.data_access
    parent_id = local_settings.parent_location_id

    data.locations = location_hierarchy(
        data_access.gbd_round_id, location_set_version_id=data_access.location_set_version_id)
    children = get_descendants(data.locations, parent_id, children_only=True)
    if not children:
        children = [parent_id]
    data_cnt = 100
    seqs = sorted(rng.choice(10 * data_cnt, size=data_cnt, replace=False))
    data.bundle = pd.DataFrame(dict(
        seq=seqs,
        measure=np.repeat(["prevalence", "Sincidence", "mtother", "mtexcess"], 25),
        mean=np.repeat([0.01, 0.02, 0.03, 0.04], 25),
        sex_id=np.repeat([1, 2, 3, 2], 25),
        lower=np.repeat([0.005, 0.003, 0.004, 0.002], 25),
        upper=np.repeat([0.015, 0.023, 0.034, 0.0042], 25),
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
        data.sparse_covariate_data = pd.DataFrame(dict(
            study_covariate_id=pd.Series(dtype=np.dtype("int64")),
            seq=pd.Series(dtype=np.dtype("O")),
            bundle_id=pd.Series(dtype=np.dtype("int64")),
        ))

    country_ids = list(set(ct_set.country_covariate_id for ct_set in local_settings.settings.country_covariate))
    age_group = [1] + list(range(6, 22))
    fake_country_cov = pd.DataFrame([dict(
        age_lower=float(5 * age_id),
        age_upper=float(5 * (age_id + 1)),
        time_lower=float(year),
        time_upper=float(year + 1),
        mean_value=rng.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
        sex_id=sex,
    )
        for age_id in range(len(age_group))
        for year in range(1990, 2018)
        for sex in [1, 2]
    ])
    data.country_covariates = {cid: fake_country_cov for cid in country_ids}
    data.country_covariates_binary = {bid: False for bid in country_ids}

    data.ages_df = db_queries.get_age_metadata(
        age_group_set_id=data_access.age_group_set_id,
        gbd_round_id=data_access.gbd_round_id
    )
    data.years_df = db_queries.get_demographics(
        gbd_team="epi", gbd_round_id=data_access.gbd_round_id)["year_id"]

    include_birth_prevalence = local_settings.settings.model.birth_prev
    data.average_integrand_cases = \
        make_average_integrand_cases_from_gbd(
            data.ages_df, data.years_df, local_settings.sexes,
            local_settings.children, include_birth_prevalence)

    parent_and_children = [local_settings.parent_location_id] + children
    locations_for_asdr = all_locations_with_these_parents(
        data.locations, parent_and_children
    )
    data.cause_specific_mortality_rate = fake_csmr(
        locations_for_asdr, local_settings.sexes, data.ages_df
    )
    data.age_specific_death_rate = fake_asdr(
        locations_for_asdr, local_settings.sexes, data.ages_df
    )
    data.study_id_to_name, data.country_id_to_name = find_covariate_names(
        execution_context, covariate_data_spec)
    assign_epiviz_covariate_names(
        data.study_id_to_name, data.country_id_to_name, covariate_data_spec
    )

    # These are the draws as output of the parent location.
    data.draws = None

    # The parent can also supply integrands as a kind of prior.
    # These will be shaped like input measurement data.
    data.integrands = None
    columns_wrong = validate_input_data_types(data)
    assert not columns_wrong, f"validation failed {columns_wrong}"
    return data
