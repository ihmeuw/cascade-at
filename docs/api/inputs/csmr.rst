.. _cause-specific-mortality-rate:

Cause-Specific Mortality Rate
=============================

The cause-specific mortality rate (CSMR) is mortality rate for one disease
by age group ID. This comes from ``db_queries.get_outputs``.
It is saved in Tier 3 in order to keep the exact version used for a run.
It is retrieved for

Where the mean of the CSMR is null, the value is dropped.
The kept columns are year, location, sex, age, mean (called val),
upper, and lower.


.. literalinclude:: ../../src/cascade_at/inputs/csmr.py
   :pyobject: CSMR
