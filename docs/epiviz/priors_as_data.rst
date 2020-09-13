.. _epiviz-priors-input:

Priors Input
============

.. _epiviz-priors-as-data:

Priors as Data
--------------

.. warning::

    This is not implemented.

As described in :ref:`epiviz-posteriors`, a single estimation can create
integrands for child locations and make point estimates for the value
and standard error of those child values. These become input for the
child estimation and are a way to specify prior belief by adding to input
data. They are added to the bundle as though they came from the bundle.
If the parent fit did not include random effects, then the input data
will be from the fit at the parent level.

The `minimum measured CV option <https://bradbell.github.io/dismod_at/doc/option_table.htm#minimum_meas_cv>`_
in Dismod-AT, represented in the Cascade UI on the Advanced tab,
will apply to this data as it applies to all data in the bundle.

There is an option in the Cascade user interface
to limit which of the integrands from the parent to include in the data.

.. attention::

    Does the min cv field in the user interface refer to the data priors,
    to the fit priors, or to both? The advanced tab sets a min CV on the
    data, and that will apply to this prior data, too. That value can
    be `set by integrand <https://bradbell.github.io/dismod_at/doc/integrand_table.htm>`_,
    but not in the EpiViz-AT user interface.


Priors on Fit
-------------

.. warning::

    This is not implemented.


As described in :ref:`posteriors-to-priors`, each single estimation can
use the set of fits, created from simulation draws, in order to create
priors.
Because these are priors on the fit, they are priors on only the five
rates, iota, rho, chi, omega, and pini. We get those priors by looking
at the five principal integrands (incidence, remission, excess mortality,
other-cause mortality and initial prevalence) at point values.

However, the priors on fit apply to

 *  primary rates - the value priors, the dage priors, the dtime priors,
    and the mulstd priors, if present. These values can be calculated from
    fits or calculate by the
    `Dismod-AT predict command <https://bradbell.github.io/dismod_at/doc/predict_command.htm>`_.

 *  *not* the random effects

 *  covariate multipliers on rate measurements, data measurements, and
    data standard deviations (alpha, beta and gamma covariates).
    There are value priors, dage priors, and dtime priors for each of
    these covariate multipliers. These
    priors *cannot come from the Dismod-AT predict function* because it
    does not predict values for the covariate multipliers.

.. note::

    Do we set priors on all of value, dage, and dtime, or some subset of these?
    Do we set primary rates on the child using the parent underlying
    rate and parent's random effect for the child, or do we also include
    the covariate multipliers when setting the child underlying rates?

Input data for priors on fit has a different shape from priors as input
data points. If a grid in the parent estimation has a different shape
from a grid in the child estimation, then the ages and times that
define the knots, or support, of the parent grid will differ from ages
and times for the child prior locations.
It will have the following columns.

 *  ``location_id`` - A child location ID if the parent had random effects.
    If the parent estimated using only fixed effects, this is the parent
    location ID.

 *  ``age_lower`` and ``age_upper`` - These will be the same value. This is
    a point estimation. Input data ages will be the knots used by the parent.
    These completely define the fit as a function of age and time.

 *  ``time_lower`` and ``time_upper`` - These will be the same value.
    Input data times will be the knots used by the parent.

 *  ``sex_id`` - For which sex is this the fit. Assuming male or female as 1 or 2.

 *  ``grid_name`` - This is an identifier for a particular smooth grid
    in the parent estimation. It can be a random effect for this child or
    a covariate multiplier's identifier. Uncertain what this is.

 *  ``draw`` - Store all of the draws because summarizing won't work if the
    child smooth grid knots differ from those in the parent.

 *  ``mean`` - The actual value for this draw, age, and time point.

This ends up being quite different from the priors-as-data described above.
Think of each draw from the parent estimation as a function :math:`f_d(a, t)`,
where :math:`d` is the index of the draw. It's a continuous function determined
by values at the knots.
This step does an MLE by evaluating that function at some possibly-new
age-time point :math:`f_d(a_j, t_j)`.

.. note::

    Do the "rate CV" from the user interface apply to the priors on the grids?
    How do we determine precedence between fit by level and fit by integrand?
