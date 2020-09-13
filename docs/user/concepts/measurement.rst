.. _dismod-measurement:

Measurement, Rate, Integrand
----------------------------

The Dismod-AT program has
`its own documentation <https://bradbell.github.io/dismod_at/doc/dismod_at.htm>`_,
which serves well for specifics about database tables, definitions of distributions,
and other details. This documentation is a high-level view of what Dismod-AT does
in order to explain what you can do with the Cascade.

Dismod-AT does statistical estimation. It is a nonlinear, multi-level regression.
The two hierarchical levels are the measurements, at the micro level, and the locations,
at the macro level.

.. index:: measurement

*Measurements* are input data from data bundles. Every measurement has
a positive, non-zero standard deviation. A measurement may or may not have
the same upper and lower age or the same upper and lower time.
All measurements are associated with locations.

Dismod-AT's central feature is that it estimates rates of a disease process.
The disease process is nonlinear and described by a
`differential equation <https://bradbell.github.io/dismod_at/doc/avg_integrand.htm>`_.
We can discuss the behavior of that model in detail later. For this differential
equation,

 1. Rates go in.
 2. Prevalence and death comes out.

.. index:: rate

A *Rate* is incidence, remission, excess mortality, other-cause mortality, or
initial prevalence. A rate is a continuous function of age and time. It's specified
as a set of points, and interpolated between those points, but it's continuous.
Even the initial prevalence is continuous across time but defined only for the youngest
age. The data associated with rates is defined at points of age and time, so
it isn't associated with age or time ranges. It also doesn't have standard
deviations.

If we think of a typical linear regression,

.. math::

    y = a + bx + \epsilon

we can draw an equivalence for Dismod-AT where :math:`x` are the covariates,
:math:`b` are the covariate multipliers, :math:`\epsilon` are distributions of priors,
`a` are the rates, and `y` are the observations. How Dismod-AT connects
rates to observations is much more complicated than a typical linear regression.

In order to relate a rate to an observation, Dismod-AT has to do a
few steps.

 1. Use the ODE to predict prevalence and death.
 2. Construct a function of rates, prevalence, and death to form the desired
    observation.
 3. Integrate that function over the requested age and time range to get a single
    value for the observation.

.. index:: integrand

Integrands are *outputs* from Dismod-AT that are predictions of either
measurements or rates.
Because studies observe participants with ranges of ages over periods of time,
they are generally associated with the integral of the continuous rates
underlying the disease process. For this reason, Dismod-AT
calls its predictions of observations *integrands.* It supports a wide
`variety of integrands <https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Integrand,%20I_i(a,t)>`_.

