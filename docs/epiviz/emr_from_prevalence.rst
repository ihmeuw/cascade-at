.. _emr-from-prevalence:

Excess Mortality Rate from Prevalence
=====================================

Dismod-AT's estimation can behave better when data is
one of the five primary integrands: S-incidence, remission,
excess mortality rate, other-cause mortality rate, or
initial prevalence at birth. Therefore, there is a method
to impute excess mortality rate from prevalence.

*This is not currently enabled in the code, as defined by
the policies section of form.py.*

.. literalinclude:: ../../src/cascade/input_data/emr.py
   :pyobject: _calculate_emr_from_csmr_and_prevalence
