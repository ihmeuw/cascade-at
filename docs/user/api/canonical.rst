.. _canonical-names:

Constants
---------

Dismod-AT makes assumptions about the order of variables.
In some cases, it has relaxed those assumptions over time,
but we retain these as conventions.

.. autoclass:: cascade.dismod.constants.RateEnum
   :members:

.. autoclass:: cascade.dismod.constants.IntegrandEnum
   :members:

.. autoclass:: cascade.dismod.constants.DensityEnum
   :members:

.. autoclass:: cascade.dismod.constants.WeightEnum
   :members:

.. autoattribute:: cascade.dismod.constants.INTEGRAND_TO_WEIGHT

   Each integrand has a natural association with a particular weight because
   it is a count of events with one of four denominators: constant, susceptibles,
   with-condition, or the total population.

.. autoattribute:: cascade.dismod.constants.INTEGRAND_COHORT_COST

    If a value is True, then rate covariates on this integrand type are much
    more costly than those that are False.
    If a value is True, then for a data point of this integrand type,
    if it has a rate covariate that is not equal to its reference value, then
    the differential equation needs to be solved for each cohort in the
    ODE grid that intersects or surrounds the age-time span for this point. This
    computation is performed each time the model variables change.
