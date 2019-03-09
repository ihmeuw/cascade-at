.. _age-specific-death-rate:

Age-Specific Death Rate
=======================

The age-specific death rate (ASDR) is total mortality rate
by age group ID. This comes from ``db_queries.get_envelope``.
It is retrieved for

 *  All locations
 *  All years
 *  Current GBD round ID.
 *  Both sexes
 *  **With-HIV=True**
 *  As rates, not counts.

Where the mean of the ASDR is null, the value is dropped.

.. literalinclude:: ../../src/cascade/input_data/db/asdr.py
   :pyobject: get_asdr_data
