.. _data-eta-cv:

Data Eta and Coefficient of Variation
=====================================

There are two modifications to the input data that are applied
before running a model.

The data eta is a setting in EpiViz-AT that sets the value
:math:`\eta` on all priors. This value changes calculation of
residuals near zero.

*The minimum cv setting in EpiViz-AT is currently ignored.*
It is set to a dummy setting in form.py.
This permits setting a min cv for each integrand. It sets
the uncertainty for a measured data value to a set fraction
of the input mean value, or to the actual value, whichever
is greater.
