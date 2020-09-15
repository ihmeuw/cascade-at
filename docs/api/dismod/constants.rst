.. _constants:

Constants
---------

Dismod-AT makes assumptions about the order of variables.
In some cases, it has relaxed those assumptions over time,
but we retain these as conventions.

.. autoclass:: cascade_at.dismod.constants.RateEnum
   :members:

.. autoclass:: cascade_at.dismod.constants.IntegrandEnum
   :members:

.. autoclass:: cascade_at.dismod.constants.DensityEnum
   :members:

.. autoclass:: cascade_at.dismod.constants.WeightEnum
   :members:

.. autoattribute:: cascade_at.dismod.constants.INTEGRAND_TO_WEIGHT

   Each integrand has a natural association with a particular weight because
   it is a count of events with one of four denominators: constant, susceptibles,
   with-condition, or the total population.

