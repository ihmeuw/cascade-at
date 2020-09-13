.. _dismod-model-variables:

Model Variables - The Unknowns
------------------------------

When we ask Dismod-AT to do a fit, what unknowns will it solve for?
If we do a fit to a linear regression, :math:`y ~ b_0 + b_1 x`,
then it tells us the parameters :math:`b_i`. It also tells us
the uncertainty, as determined by residuals between predicted and
actual :math:`y`. In the case of Dismod-AT, the model variables are
equivalent to those parameters :math:`b_i`.
Dismod-AT documentation lists all of the
`model variables <https://bradbell.github.io/dismod_at/doc/model_variables.htm>`_, but
let's cover the most common ones here.

First are the five disease rates, which are inputs to the ODE. Each rate is
a continuous function of age and time, specified by an interpolation among points
on an age-time grid. Therefore, the model variables from a rate are its value
at each of the age-time points.

The covariate multipliers are also continuous functions of age and time.
Each of the covariate multipliers has model variables for every point in its
smoothing. There can be a covariate multiplier for each combination of
covariate column and application to rate value, measurement value, or measurement
standard deviation, so that's a possible :math:`3c` covariate multipliers, where
:math:`c` is the number of covariate columns.

The child rate effects also are variables. Because there is one for each location,
and there is a smoothing grid for child rate effects, this creates many model variables.
