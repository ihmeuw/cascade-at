.. _cause-specific-mortality-rate:

Cause-Specific Mortality Rate
=============================

The cause-specific mortality rate (CSMR) is mortality rate for one disease
by age group ID. This comes from ``db_queries.get_outputs``.
It is saved in Tier 3 in order to keep the exact version used for a run.
It is retrieved for

 *  The cause requested, which can differ from the cause under study
    as requested in EpiViz-AT.
 *  All locations.
 *  The metric for  per-capita rate.
 *  All years
 *  "most detailed" age groups.
 *  Measure ID for deaths.
 *  All sexes. Neither is treated as both.
 *  Current GBD round ID.
 *  The process version, for gbd process id 3, metadata type id 1,
    gbd process version status 1, and cod version.

Where the mean of the CSMR is null, the value is dropped.
The kept columns are year, location, sex, age, mean (called val),
upper, and lower.


.. literalinclude:: ../../src/cascade/input_data/db/csmr.py
   :pyobject: get_csmr_data
