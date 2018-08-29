.. _dismod-description:

Dismod-AT Concepts
==================

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


.. _dismod-command-flow:

Flow of Commands in Dismod-AT
-----------------------------

There are a few different ways to use Dismod-AT to examine data.
The correspond to different sequences of
`Dismod-AT commands <https://bradbell.github.io/dismod_at/doc/command.htm>`_.

.. _stream-out-prevalence:

**Stream Out Prevalence** The simplest use of Dismod-AT is to ask it to run the ODE on known
rates and produce prevalence, death, and integrands derived from these.

  1. *Precondition* Provide known values for all rates over the whole
     domain. List the integrands desired for the output.

  2. Run *predict* on those rates.

  3. *Postcondition* Dismod-AT places any requested integrands in
     its predict table. These can be rates, prevalence, death, or
     any of the integrands.

.. _fit-and-predict:

**Simple Fit to a Dataset** This describes a fit with the simplest way to determine
uncertainty.

  1. *Precondition* The input data is observations, with standard deviations,
     of any of the known integrands.

  2. Run *fit* on those observations to produce rates and covariate multipliers.

  3. Run *predict* on the rates to produce integrands.

  4. *Postcondition* Integrands are in the predict table.

.. _fit-asymptotic:

**Fit with Asymptotic Uncertainty** This fit produces some values of uncertainty.

  1. *Precondition* The input data is observations, with standard deviations,
     of any of the known integrands.

  2. Run *fit* on those observations to produce rates and covariate multipliers.

  3. Run *sample asymptotic.*

  4. *Postcondition* Integrands are in the predict table.

.. _fit-simulate:

**Fit with Simulated Uncertainty** This uses multiple predictions in order
to obtain a better estimate of uncertainty.

  1. *Precondition* The input data is observations, with standard deviations,
     of any of the known integrands.

  2. Run *fit* on those observations to produce rates and covariate multipliers.

  3. Run *simulate* to generate simulations of measurements data and priors.

  4. Run *sample simulate.*

  5. *Postcondition* Integrands are in the predict table.


.. _dismod-smoothing:

Smoothing Continuous Functions
------------------------------

We said that rates and :ref:`covariate multipliers <covariates>` are continuous functions of age and time.
It takes a little work to parametrize an interpolated function of age and time.

 * You have to tell it where the control points are. In Cascade, we call this
   the :py:class:`AgeTimeGrid <cascade.model.grids.AgeTimeGrid>`.
   It's a list of ages and a list of times
   that define a rectangular grid.

 * At each of the control points of the age time grid, Dismod-AT will evaluate
   how close the rate or covariate multiplier is to some reference value. At these
   points, we define prior distributions. Cascade makes these *value priors*
   part of the :py:class:`PriorGrid <cascade.model.grids.PriorGrid>`.

 * It's rare to have data points that are dense across all of age and time.
   Dismod-AT needs to take a data point at one end, a data point at the other
   end, and draw a line that connects them. We help it by introducing constraints
   on how quickly a value can change over age and time. These are a kind of
   regularization of the problem, called *age-time difference priors*. They apply
   to the difference in value between one age-time point and the next greater
   in age and the next-greater in time. As with value priors, these are specified
   in the Cascade as part of the :py:class:`PriorGrid <cascade.model.grids.PriorGrid>`.

The random effect for locations is also a continous quantity.

.. _dismod-hierarchical:

Hierarchical Model
------------------

The hierarchical part of Dismod-AT does one thing, estimate how
locations affect rates. If the rate at grid point :math:`(i,k)`
is :math:`q_{ik}(a,t)`, and the covariate
multiplier is :math:`\alpha_{ik}(a,t)`, then the adjusted rate is

.. math::

    r_{ik}(a,t) = q_{ik}(a,t) \exp\left(u_{ik}(a,t) + \sum_j x_{ikj}\alpha_{jik}(a,t)\right).

The offset, :math:`u`, is linear with the covariates, but it is inside the exponential,
which guarantees that all rates remain positive. This offset is the only
random effect in the problem, and it is called the
*child rate effect* because each location, or node in Dismod-AT's language,
is considered a child of a parent.

Because the child rate effect is continuous, you can conclude that it must be
defined on a smoothing grid. Dismod-AT will either define one smoothing grid
for each child rate effect (one for each of the five rates) or let you define
a smoothing grid for every location and every child rate effect, should that
be necessary.

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

The covariate multipliers also continuous functions of age and time.
Each of the covariate multipliers has model variables for every point in its
smoothing. There can be a covariate multiplier for each combination of
covariate column and application to rate value, measurement value, or measurement
standard deviation, so that's a possible :math:`3c` covariate multipliers, where
:math:`c` is the number of covariate columns.

The child rate effects also are variables. Because there is one for each location,
and there is a smoothing grid for child rate effects, this creates many model variables.
