"""
This example constructs makes a test disease model, similar to
diabetes, and sends that data to DismodAT.

1.  The test model is constructed by specifying functions for
    the primary rates, incidence, remission, excess mortality rate,
    and total mortality. Then this is solved to get prevalence over time.

2.  Then construct demographic observations of these rates. This means
    averaging over the rates for given ages and times.

3.  Load a subset of these observations into a DismodAT file and run
    DismodAT on it.

This example works in cohort time, so that rates don't change over
years.

.. autofunction: pretend_diabetes


.. autofunction: observe_demographic_rates

"""
import logging
from pathlib import Path
import pickle
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy.stats import gamma

from cascade.model import (
    siler_default,
    prevalence_solution,
    integrands_from_function,
    omega_from_mu,
    DemographicInterval,
)
from cascade.input_data.configuration.construct_bundle import \
    normalized_bundle_from_database
from cascade.dismod.constants import IntegrandEnum

LOGGER = logging.getLogger("fit_no_covariates")


def pretend_diabetes():
    """
    Create theoretical distributions for a disease. Make just enough to
    specify the problem. Then let the library calculate what those imply.
    In this case, we loaded diabetes data from the United States in 2015
    and made, by hand, rates of disease that sort-of match the age pattern
    of diabetes in that year.
    """

    def diabetes_incidence(x):
        return 0.8 * gamma(a=9, scale=7).pdf(x)

    def diabetes_emr(x):
        """Create an excess mortality rate."""
        return 0.015 * (np.exp((x / 100) ** 2) - 1)

    def diabetes_remission(x):
        return np.zeros_like(x)

    total_mortality = siler_default()

    # Use those rates to parameterize a differential equation, which we then
    # solve to get susceptibles, with-condition, and prevalence = S/(S+C)
    # as functions of cohort time.
    S, C, P = prevalence_solution(diabetes_incidence, diabetes_remission, diabetes_emr, total_mortality)

    # We need N=S+C, which is the number of people alive. It's also known
    # as :math:`l_x(t)` in demography terms. This will be necessary to
    # create observations on a theoretical population.
    def lx(t):
        return S(t) + C(t)

    omega = omega_from_mu(total_mortality, diabetes_emr, P)

    return dict(
        incidence=diabetes_incidence,
        emr=diabetes_emr,
        remission=diabetes_remission,
        total_mortality=total_mortality,
        other_mortality=omega,
        susceptible=S,
        with_condition=C,
        prevalence=P,
        lx=lx,
    )


def observe_demographic_rates(rates, ages):
    """
    Given continuous functions, return rates as averages over intervals,
    done in the same way they would be observed for populations. This means
    they are averaged over :math:`l_x`, the number of people alive across
    the interval.

    Args:
        rates (dict[str,function]): A list of functions of cohort-time, defined
            for all times in the age intervals.
        ages (DemographicInterval): Age intervals for which to predict
            results.

    Returns:
        list(np.array): A list of estimated integrands on the age intervals.
    """
    rates, norm = integrands_from_function([rates["incidence"], rates["with_condition"]], rates["lx"], ages)
    return rates + [norm]


def theory():
    rates = pretend_diabetes()
    intervals = DemographicInterval(np.full((10,), 10.0, dtype=np.float))
    observations = observe_demographic_rates(rates, intervals)
    return observations


def cached_bundle_load(context, bundle_id, tier_idx):
    cache_bundle = Path(f"{bundle_id}.pkl")
    if cache_bundle.exists():
        LOGGER.info(f"Reading bundle from {cache_bundle}. " f"If you want to get a fresh copy, delete this file.")
        return pickle.load(cache_bundle.open("rb"))

    LOGGER.debug(f"Begin getting study covariates {bundle_id}")
    bundle_begin = timer()
    bundle = normalized_bundle_from_database(context, bundle_id, tier_idx)
    LOGGER.debug(f"bundle is {bundle} time {timer() - bundle_begin}")

    pickle.dump((bundle, covariate), cache_bundle.open("wb"), pickle.HIGHEST_PROTOCOL)
    return bundle, covariate


def bundle_to_observations(bundle_df):
    for check_measure in bundle_df["measure"].unique():
        if check_measure not in IntegrandEnum.__members__:
            raise KeyError(f"{check_measure} isn't a name known to Cascade.")

    if "location_id" in bundle_df.columns:
        location_id = bundle_df["location_id"]
    else:
        location_id = np.full(len(bundle_df), -1, dtype=np.int)
    return pd.DataFrame(
        {
            "measure": bundle_df["measure"],
            "location_id": location_id,
            "age_start": bundle_df["age_start"],
            "age_end": bundle_df["age_end"],
            "year_start": bundle_df["year_start"].astype(np.float),
            "year_end": bundle_df["year_end"].astype(np.float),
            "mean": bundle_df["mean"],
            "standard_error": bundle_df["standard_error"],
        }
    )


def observations_to_integrand(bundle_df, node_id):
    measure_to_integrand = dict(incidence=IntegrandEnum.Sincidence.value, mtexcess=IntegrandEnum.mtexcess.value)
    return pd.DataFrame(
        {
            "measure": bundle_df["measure"].apply(measure_to_integrand.get),
            "node_id": bundle_df["location_id"],
            "age_lower": bundle_df["age_start"],
            "age_upper": bundle_df["age_end"],
            "time_lower": bundle_df["year_start"].astype(np.float),
            "time_upper": bundle_df["year_end"],
            "meas_value": bundle_df["mean"],
            "meas_std": bundle_df["standard_error"],
            "hold_out": [0] * len(bundle_df),
        }
    )
