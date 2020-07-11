from functools import lru_cache
import pandas as pd

from cascade_at.dismod.constants import IntegrandEnum
from cascade_at.core.log import get_loggers

LOG = get_loggers(__name__)


RATE_TO_INTEGRAND = {
    "iota": IntegrandEnum.Sincidence,
    "rho": IntegrandEnum.remission,
    "chi": IntegrandEnum.mtexcess,
    "omega": IntegrandEnum.mtother,
    "pini": IntegrandEnum.prevalence
}


PRIMARY_INTEGRANDS_TO_RATES = {
    "prevalence": "pini",
    "Sincidence": "iota",
    "Tincidence": "iota",
    "incidence": "iota",
    "remission": "rho",
    "mtexcess": "chi",
    "mtother": "omega"
}


INTEGRAND_ENCODED = """
idx measure_id                                     measure_name
0            1                                           Deaths
1            2           DALYs (Disability-Adjusted Life Years)
2            3               YLDs (Years Lived with Disability)
3            4                        YLLs (Years of Life Lost)
4            5                                       Prevalence prevalence
5            6                                        Incidence incidence
6            7                                        Remission remission
7            8                                         Duration
8            9                            Excess mortality rate mtexcess
9           10               Prevalence * excess mortality rate
10          11                                    Relative risk relrisk
11          12                     Standardized mortality ratio mtstandard
12          13                    With-condition mortality rate mtwith
13          14                         All-cause mortality rate mtall
14          15                    Cause-specific mortality rate mtspecific
15          16                       Other cause mortality rate mtother
16          17                               Case fatality rate
17          18                                       Proportion
18          19                                       Continuous
19          20                                    Survival Rate
20          21                                Disability Weight
21          22                               Chronic Prevalence
22          23                                 Acute Prevalence
23          24                                  Acute Incidence
24          25                         Maternal mortality ratio
25          26                                  Life expectancy
26          27                             Probability of death
27          28                   HALE (Healthy life expectancy)
28          29                           Summary exposure value
29          30                Life expectancy no-shock hiv free
30          31                Life expectancy no-shock with hiv
31          32           Probability of death no-shock hiv free
32          33           Probability of death no-shock with hiv
33          34                                   Mortality risk
34          35                            Short term prevalence
35          36                             Long term prevalence
36          37           Life expectancy decomposition by cause
37          38                                 Birth prevalence prevalence
38          39                  Susceptible population fraction susceptible
39          40               With Condition population fraction withC
40          41                            Susceptible incidence Sincidence
41          42                                  Total incidence Tincidence
42          43  HAQ Index (Healthcare Access and Quality Index)
43          44                                       Population
44          45                                        Fertility
"""
"""Generated with
   from db_queries import get_ids
   get_ids("measure")
We do it this way to make it as easy as possible to check.
This maps Incidence to Tincidence because the decision to forbid it
happens when decoding the data, not here.
"""


def make_integrand_map():
    """Makes dict where key=GBD measure_id, value=IntegrandEnum member"""
    split_column = 64
    mapp = {int(line.split()[1]): IntegrandEnum[line[split_column:].strip()]
            for line in INTEGRAND_ENCODED.splitlines()
            if len(line) > split_column}
    return mapp


INTEGRAND_MAP = make_integrand_map()


"""From Dismod integrand to Dismod primary rate name"""


def reverse_integrand_map():
    """
    Makes a dictionary where key=integrand_name, value=GBD measure_id

    NOTE: Over-rides the birth prevalence measure because birth prevalence is
    defined by age_group_id in IHME databases, not measure in DisMod.
    """
    mapping = {v.name: k for k, v in make_integrand_map().items()}
    mapping['prevalence'] = 5
    return mapping


def integrand_to_gbd_measures(df: pd.DataFrame, integrand_col: str) -> pd.DataFrame:
    """
    Maps the integrand column to measure IDs
    and adds in filler measures where necessary (e.g. copies
    over Sincidence to incidence).

    Parameters
    ----------
    df
        data frame with integrand
    integrand_col
        column name for integrand column

    Returns
    -------
    data frame with integrands mapped to measures
    """
    integrand_map = reverse_integrand_map()
    df['measure_id'] = df[integrand_col].apply(lambda x: integrand_map[x])

    # Duplicates the Sincidence results, if they exist, so that they
    # show up as incidence in the visualization tool
    if integrand_map['Sincidence'] in df.measure_id.unique():
        incidence = df.loc[df.measure_id == integrand_map['Sincidence']].copy()
        incidence['measure_id'] = integrand_map['incidence']
        df = pd.concat([df, incidence], axis=0).reset_index()
    return df
