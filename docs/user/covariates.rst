.. _covariates:

Covariates
==========

Covariates are the independent variables in the statistical
model. They appear as columns in observation data, associated
with each measurement. The word covariate is overdetermined,
so we will refer to a covariate column, a covariate use,
a covariate multiplier, and applying a covariate.

A covariate column has a unique name and a reference value
for which the observed data is considered unadjusted.
All priors on covariates are with respect to this
unadjusted value.


Covariate Uses and Applications
-------------------------------

There are three reasons to use a covariate.

*Country Covariate*
    We believe this covariate predicts disease behavior.

*Study Covariate*
    The covariate marks a set of studies that behave differently.
    For instance, that set may have different criteria for incipience
    of a disease. We assign a covariate to the set of studies
    to account for bias from study design.

*Sex Covariate*
    This is usually used to select a subset of data by sex,
    but this could be done based on any covariate associated
    with observation data. In addition to being used to subset
    data, the sex covariate is a covariate multiplier applied
    the same way as a study covariate.

A covariate column that is used just for exclusion doesn't need
a covariate multiplier. In practice, the sex covariate is used
at global or super-region level as a study covariate. Then the
adjustments determined at the upper level are applied as constraints
down the hierarchy. This means there is a covariate multiplier
for sex, and its smooth is a grid of constraints, not typical
priors.

Dismod-AT applies covariate effects to one of three different variables.
It either uses the covariate to `predict the underlying rate`_,
or it applies the covariate to predict the measured data. It can
be an effect on either the `measured data value`_ or the
observation data `standard deviation`_. Dismod-AT calls these
the alpha, beta, and gamma covariates.

As a rule of thumb, the three uses of covariates apply
to different variables, as shown in the table below.

====================  =======  ================ ===============
Use of Covariate      Rate     Measured Value   Measured Stddev
====================  =======  ================ ===============
Country               Yes      Maybe            Maybe
Study                 Maybe    Yes              Yes
Sex (exclusion)       No       Yes              No
====================  =======  ================ ===============

Outliering by Covariates
------------------------
Each covariate column has an optional *maximum difference*
to set. If the covariate is beyond the maximum difference from
its reference value, then the data point is outliered.
As a consequence, that data point will not be in the data
subset table. Nor will it subsequently appear in the avgint table.

Country and study covariates can optionally use outliering.
The sex covariate is defined by its use of regular outliering.
Male and female data is assigned a value of -0.5 and 0.5, and
the mean and maximum difference are adjusted to include one,
the other, or both sexes.

If there is a need to use two different references or
maximum differences for the same covariate column, then
duplicate the column.


Usage
-----

Skip for now how to obtain covariate data from IHME resources
for a given bundle. Assume the input data has covariate columns
associated with it.

In order to use a covariate column as a country covariate, specify

 * its reference value
 * an optional maximum difference, beyond which covariate
   value the data which it predicts will be considered an outlier,
 * one of the five rates (iota, rho, chi, omega, pini),
   to which it will apply
 * a smoothing grid, as a support on which the covariate effect
   is solved. This grid defines a mean prior and elastic
   priors on age and time, as usual for smoothing grids.

We give Dismod-AT measured data with associated covariates.
Dismod-AT treats the covariates as a continuous function of age
and time, which we call the *covariate multiplier.* It solves for
that continuous function, much like it solves for the rates.
Therefore, each application of a covariate column to a
rate or measured value or standard deviation requires a smoothing
grid.

Applying a study covariate is much the same, except that it
usually applies not to a rate but to the value or standard deviation
of an integrand.

For instance::

    # Assume smooth = Smooth() exists.
    income = Covariate("income", 1000)
    income_cov = CovariateMultiplier(income, smooth)

    model.rates.iota.covariate_multipliers.append(income)
    model.outputs.integrands.prevalence.value_covariate_multipliers.append(income)
    model.outputs.integrands.prevalence.std_covariate_multipliers.append(income)

Covariates are unique combinations of the covariate column,
and the rate or measured value or standard deviation,
so they can be accessed that way.


.. _predict the underlying rate:
    https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Rate%20Functions.Rate%20Covariate%20Multiplier,%20alpha_jk

.. _measured data value:
    https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Measurement%20Value%20Covariates.Multiplier,%20beta_j

.. _standard deviation:
    https://bradbell.github.io/dismod_at/doc/data_like.htm#Measurement%20Standard%20Deviation%20Covariates.gamma_j
