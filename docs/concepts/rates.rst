
.. _disease-hazard-rates:

Hazard Rates
------------
The hazard rate is defined first for an individual:

    A *hazard rate* is the probability, per unit time, that an event
    will happen given that it has not yet happened.

For a population, the hazard rate is the sum of the hazard rates for
all individuals in that population. For instance, the remission rate,
as a function of age, averages over all the different times someone
may have entered the with-condition state.

The Dismod-AT `compartmental model <https://en.wikipedia.org/wiki/Multi-compartment_model>`_
has four *Dismod-AT primary rates,*
all of which are hazard rates,

 * Susceptible Incidence rate, :math:`\iota`
 * Remission rate, :math:`\rho`
 * Excess mortality rate, :math:`\chi`
 * Other-cause mortality rate, :math:`\omega`

and an initial condition, birth prevalence, :math:`p_{ini}`.
We call the primary rates *hazard rates* because they are
the probability per unit time that an individual, age :math:`x`, moves from
one compartment to another, given that they have not yet
left their current compartment. Note that birth prevalence for a cohort
is, when we look at it across years, a birth rate. That is why
you will see birth prevalence called one of the Dismod-AT primary rates.

These primary rates are
exactly the parameters in the Dismod-AT differential equation,

.. math::

    \frac{dS(x)}{dx} = -\iota(x) S(x) +\rho(x) C(x) - \omega(x) S(x)

    \frac{C(x)}{dx} = \iota(x) S(x) - \rho(x) C(x) - \left(\omega(x) + \chi(x)\right) C(x)

where :math:`S(x)` are susceptibles as a function of cohort age and
:math:`C(x)` are with-condition as a function of cohort age.
