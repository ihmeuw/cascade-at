.. _posteriors-to-priors:

Posteriors to Priors
====================

Given the solved estimation at a location, results can be used
to prime estimation at all child locations. There are two general
techniques. One is to set parameters on prior distributions for
the child model. The other is to take predictions from the parent
and include them as data for the child.

How Cascade Sets Priors
^^^^^^^^^^^^^^^^^^^^^^^
Constructing priors is done in
:py:func:`set_priors_from_parent_draws <cascade.executor.priors_from_draws.set_priors_from_parent_draws>`.
That link leads to the code that implements this section.

If we label a grid in accordance with Dismod-AT's labeling, then
it is an age-by-time rectangular grid of values, :math:`v_{at}`.
There are value priors, dage priors, and dtime priors:

.. math::

    v_{at} \sim \mbox{dist}(\mu, \sigma)

    v_{a't} - v_{at} \sim \mbox{dist}(\mu, \sigma)

    v_{at'} - v_{at} \sim \mbox{dist}(\mu, \sigma)

where :math:`a'` is the next-larger value of :math:`a` and
:math:`t'` is the next-larger value of :math:`t`.
The distributions are chosen by the modeler, so "dist" can
be :math:`N(\mu, \sigma)` or it can be a students-t of
:math:`(\mu, \sigma, \nu)`, or any of the other distributions.
Setting priors means estimating parameters for the distributions
of the value, dage, and dtime priors from draws of a parent estimation.
For some distributions, the estimator is MLE. For those distributions
where MLE is unavailable, the mean of the draws and standard deviation
of the draws are used as estimators, as shown in
:py:mod:`cascade.model.priors`.

The relationship between the parent estimation (of parent and children)
and the child estimation (of child and grandchildren) comes from the
central model of Dismod-AT, the
`adjusted rate equation <https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Rate%20Functions.Adjusted%20Rate,%20r_ik>`_. Let's copy that here and focus on a
single rate for notational simplicity. Pick remission, :math:`\rho`,
so :math:`u_i` is the random effect for :math:`\rho` for child :math:`i`.

.. math::

    \rho_i(a,t) = \exp\left[u_i(a,t) + \sum_{j\in J(\rho)}x_{j}\alpha_j(a,t)\right]q(a,t)

The child index is :math:`i`.
There are :math:`J(\rho)` covariate multipliers against the :math:`x_{i}`
covariates. Covariates are relative to their reference value.
Suppose that this equation refers to rates covariates, and covariate multipliers
for the child estimation. Write the same equation for the parent estimation,
changing indices accordingly.

.. math::

    \rho_c(a,t) = \exp\left[u_c(a,t) + \sum_{j\in J(\rho)}x'_{j}\alpha'_j(a,t)\right]q_i(a,t)

The prime on :math:`(x', \alpha')` is a reminder that we have a choice about
whether to change the reference value on covariate data, and we have another
choice about whether to allow covariate multipliers to change between parent
and child. These choices are decided, respectively, in the country covariate
construction and in the :py:class:`CascadePlan <cascade.executor.cascade_plan.CascadePlan>`.

The code equates

.. math::

    q_i(a,t) = \rho_i(a,t) \exp u_i(a,t),

so that, for each value in the grid for :math:`\rho`, the distribution
for the grid values, dage, and dtime priors comes from

.. math::

   v_{at} \sim \mbox{MLE}(\rho_i(a,t) \exp u_i(a,t))

   v_{a't} - v_{at} \sim \mbox{MLE}(\rho_i(a',t) \exp u_i(a',t) - \rho_i(a,t) \exp u_i(a,t))

   v_{at'} - v_{at'} \sim \mbox{MLE}(\rho_i(a,t') \exp u_i(a,t') - \rho_i(a,t) \exp u_i(a,t)).

As described above, Gaussian distributions do use MLE, but other distributions
may use simpler estimators, depending on what's available in Scipy.

Meanwhile, all covariates estimate value, dage, and dtime priors directly from
draws at the parent level, and random effects for grandchildren use priors
supplied by the modeler, without additional prior information.
