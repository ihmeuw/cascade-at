.. _mtother:

Other-Cause Mortality
^^^^^^^^^^^^^^^^^^^^^

The IHME databases supply all-cause mortality, but Dismod-AT
uses other-cause mortality. It can impute what it needs to know
using all-cause mortality, but it is helpful to add other-cause
mortality not just as input data but as a constraint to the model.

*We use total mortality as other-cause mortality.*
The correct formulae to use are for "cause-deleted lifetables"
or "cause deletion."

Omega Constraint
""""""""""""""""

This constrains other-cause mortality using data
from mtother, which is the integrand for other-cause mortality.

The choice to use an omega constraint is set in EpiViz-AT,
and this is obeyed. If the user does choose to constrain omega,
then it is included with the following function.

.. autofunction:: cascade_at.inputs.utilities.data.calculate_omega
