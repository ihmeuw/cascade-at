.. _posteriors-to-priors:

Posteriors to Priors
====================

The specification below was Drew's first attempt.
Brad Bell has created a
`user cascade example <https://bradbell.github.io/dismod_at/doc/user_cascade.py.htm>`_
which serves as a specification for a next attempt, from
which we will make another implementation.

Given the solved estimation at a location, results can be used
to prime estimation at all child locations. There are two general
techniques. One is to set parameters on prior distributions for
the child model. The other is to take predictions from the parent
and include them as data for the child.

Constructing priors is done in
:py:func:`set_priors_from_parent_draws <cascade.executor.priors_from_draws.set_priors_from_parent_draws>`.
That link leads to the code that implements this section.

If we label a grid in accordance with Dismod-AT's labeling, then
it is an age-by-time rectangular grid of values, :math:`v_{at}`.
There are value priors, dage priors, and dtime priors:

.. math::
    :label: prior-kind-definitions

    v_{at} \sim \mbox{dist}(\mu, \sigma)& \qquad\mbox{value prior}

    v_{a't} - v_{at} \sim \mbox{dist}(\mu, \sigma)& \qquad\mbox{dage prior}

    v_{at'} - v_{at} \sim \mbox{dist}(\mu, \sigma)& \qquad\mbox{dtime prior}

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

The relationship between a grandparent estimation (of grandparent and parent)
and a parent estimation (of parent and children) comes from the
central model of Dismod-AT, the
`adjusted rate equation <https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Rate%20Functions.Adjusted%20Rate,%20r_ik>`_.
Let's copy that here, with no changes,

.. math::
    :label: adjusted-rate-definition

    r_{i,k}(a,t) = \exp\left[u_{i,k}(a,t) + \sum_{j\in J(k)}x_{i,j}\alpha_{j,k}(a,t)\right]q_{i,k}(a,t)

and rewrite it for a
single rate for notational simplicity. Pick remission, :math:`\rho`,
so :math:`u_i` is the random effect for :math:`\rho` for child :math:`i`.
Write this same equation for the parent estimation,
so this is grandparent-to-parent, where parent's index is :math:`c`,
changing indices accordingly.

.. math::
    :label: remission-grandparent

    \rho_c(a,t) = \exp\left[u_c(a,t) + \sum_{j\in J(\rho)}(x_{j}-x_{j0})\alpha_j(a,t)\right]q_i(a,t)

There are :math:`J(\rho)` covariate multipliers against the :math:`x_{j}`
covariates. We've added a reference value, :math:`x_{j0}` for each covariate,
so that it's visible in the equation. Covariates are relative to their reference value.
We then write the parent estimation, the one for parent-to-child, **in order to ask
which terms in the grandparent estimation correspond to which terms
in the parent estimation.** So this is for parent-to-child,

.. math::
    :label: remission-parent

    \rho_i(a,t) = \exp\left[u_i(a,t) + \sum_{j\in J(\rho)}(x'_{j} - x'_{j0})\alpha'_j(a,t)\right]q(a,t).

The child index is :math:`i`. If we set :math:`u_i=0`, then we get a prediction
for the underlying rate, of the parent,

.. math::
    :label: remission-from-parent

    \rho_u = \exp\left[\sum_{j\in J(\rho)}(x'_{j} - x'_{j0})\alpha'_j(a,t)\right]q(a,t)

The prime on :math:`(x', x'_0, \alpha')` is a reminder that we have a choice about
whether to change the reference value on covariate data, and we have another
choice about whether to allow covariate multipliers to change between parent
and child.

Now that notation is set, we need to decide which terms in the grandparent-to-parent
are equal to the terms in parent-to-child. The code equates the predicted
real rate for the parent, by the grandparent-to-parent estimation, with the
predicted underlying rate for the parent-to-child estimation,

.. math::
    :label: remission-equivalence

    \rho_c(a,t) = \rho_u

.. math::
    :label: two-sided-equivalence

    \exp\left[u_c(a,t) + \sum_{j\in J(\rho)}(x_{j}-x_{j0})\alpha_j(a,t)\right]q_i(a,t) = \exp\left[\sum_{j\in J(\rho)}(x'_{j} - x'_{j0})\alpha'_j(a,t)\right]q(a,t).

Canceling on both sides, for :math:`(x', x'_0, \alpha')=(x, x_0, \alpha)` leads to,

.. math::
    :label: canceled-underlying-equivalence

    q_i(a,t)\exp\left[u_c(a,t)\right] = q(a,t),

This is the equation we use to set priors for underlying rates, the
next level down.
For each value in the grid for :math:`\rho`, the distribution
for the grid values, dage, and dtime priors comes from

.. math::
    :label: priorfromparent-value

    v_{at} \sim \mbox{MLE}(\rho_i(a,t) \exp u_c(a,t))

    v_{a't} - v_{at} \sim \mbox{MLE}(\rho_i(a',t) \exp u_c(a',t) - \rho_i(a,t) \exp u_c(a,t))

    v_{at'} - v_{at'} \sim \mbox{MLE}(\rho_i(a,t') \exp u_c(a,t') - \rho_i(a,t) \exp u_c(a,t)).

As described above, Gaussian distributions do use MLE, but other distributions
may use simpler estimators, depending on what's available in Scipy.

Meanwhile, all covariates estimate value, dage, and dtime priors directly from
draws at the parent level, and random effects for grandchildren use priors
supplied by the modeler, without additional prior information.

Code that implements this algorithms is in

 *  country covariate construction,
    :py:func:`cascade.executor.covariate_data.add_country_covariate_to_observations_and_avgints`

 *  the :py:class:`CascadePlan <cascade.executor.cascade_plan.CascadePlan>`

 *  Setting priors from draws :py:func:`cascade.executor.priors_from_draws.set_priors_from_draws`
