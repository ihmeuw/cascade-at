.. _omega-constraint:

Omega Constraint
================

This constrains other-cause mortality using data
from mtother, which is the integrand for other-cause mortality.

The choice to use an omega constraint is set in EpiViz-AT,
and this is obeyed.

We have options for how to include the omega-constraint.

 *  Include all data from all years.
 *  Subset data to years where there are measured data.
 *  Thin the data by year so it is every five years.

Adding lots of omega-constraint data can slow the program.

The current implementation is in
:py:func:`cascade.executor.construct_model.construct_model`.
In particular, :py:func:`cascade.executor.construct_model.constrain_omega`
has the math.
It creates an omega constraint using the default age and time
grid, as set in the EpiViz-AT user interface. The data used to
constrain omega is exactly
the age-specific death rate, retrieved from
``db_queries.get_envelope`` in the function
:py:func:`cascade.input_data.db.asdr.get_asdr_data`.
Further work should use "cause-deleted life tables" to impute
other-cause mortality.
