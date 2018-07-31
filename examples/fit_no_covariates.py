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
"""
import numpy as np
from scipy.stats import gamma

from cascade.model import (
    siler_default, prevalence_solution, integrands_from_function,
    DemographicInterval
)
from cascade.testing_utilities import make_execution_context


def pretend_diabetes():
    """
    Make up some distributions and turn them into integrands for Dismod.
    """
    def diabetes_incidence(x):
        return 0.8 * gamma(a=9, scale=7).pdf(x)

    def diabetes_emr(x):
        return 0.015 * (np.exp((x / 100) ** 2) - 1)

    def diabetes_remission(x):
        return np.zeros_like(x)

    total_mortality = siler_default()

    S, C, P = prevalence_solution(
        diabetes_incidence, diabetes_remission, diabetes_emr, total_mortality)

    def lx(t):
        return S(t) + C(t)

    omega = omega_from_mu(total_mortality, diabetes_emr, P)

    rates = dict(
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
    return rates


def observe_demographic_rates(rates, ages):
    rates, norm = integrands_from_function(
        [rates["incidence"], rates["with_condition"]],
        rates["lx"],
        ages,
    )
    return rates + [norm]


def construct_database():
    rates = pretend_diabetes()
    intervals = DemographicInterval(np.full((10,), 10.0, dtype=np.float))
    observations = observe_demographic_rates(rates, intervals)

    context = make_execution_context()

