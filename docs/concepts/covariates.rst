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

Outliering by Covariates
------------------------
Each covariate column has an optional *maximum difference*
to set. If the covariate is beyond the maximum difference from
its reference value, then the data point is outliered.
As a consequence, that data point will not be in the data
subset table. Nor will it subsequently appear in the avgint table.

If there is a need to use two different references or
maximum differences for the same covariate column, then
duplicate the column.


Usage
-----

Covariate data is columns in the input DataFrame and in the average
integrand DataFrame. Let's not discuss here how to obtain this covariate
data, but discuss what Dismod-AT needs to know about those
covariate columns in order to use it for a fit.

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

Missing Values
--------------

Were a covariate value to be missing, Dismod-AT would assume it has
the reference value. In this sense, *every measurement always has a covariate.*
Therefore, the interface requires every measurement explicitly have every
covariate.
