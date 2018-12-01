.. _disease-rates:

Disease Rates
=============

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

The Dismod-AT compartmental model has four *Dismod-AT primary rates,*
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



S-Incidence and T-Incidence
---------------------------

We distinguish susceptible incidence rate from total incidence rate.
These are also called s-incidence and t-incidence.
Total incidence rate is the number of new observations of a disease
per person in the population, where both people with and without the
disease are counted. Because hazard rates are the probablity per unit time
of a transition *given that the transition has not happened,*
we wouldn't call t-incidence a hazard rate because it includes people
for whom the transition to the disease state has already happened.
Both, however, can be population rates.


Population Rates
----------------

Measurements of a population count events that happen to some
set of people. They take the form

.. math::

    \frac{\mbox{number of events}}{\mbox{people exposed to that event}}

Different measurements have different denominators, and those denominators become
`weight functions in Dismod-AT <https://bradbell.github.io/dismod_at/doc/avg_integrand.htm#Weight%20Integral,%20wbar_i>`_.
If you get the weight function wrong, then you
get the comparison from hazard rates to population measurements wrong. This section
lists various measurements and their denominators.
People in the :math:`S` state are exposed to incidence and death.
People in the :math:`C` state are exposed to remission and death.

Some population rates are estimates of hazard rates. The population
rate for s-incidence is an estimate of a hazard rate. As the age-extent
and time-extent for the measurement gets closer to a point estimate,
the population rate and the hazard rate become the same value.

We can be exact about the relationship between population rates
and hazard rates by following the example of mortality rate in
Preston's *Demography.* The mortality rate is

.. math::

    {}_nm_x = \frac{\int_x^{x+n}l(a)\mu(a)da }{\int_x^{x+n}l(a)da}

where :math:`l(x)=S(x)+C(x)` is the remaining fraction of those alive and
:math:`\mu(x)` is the total mortality rate. The numerator in that equation
is the age-specific death rate and the denominator is the exposure,
as person-years lived, or :math:`{}_nL_x`. When we look at these numbers over age and time,
instead of over cohort age, :math:`x`, the integral changes to

.. math::

    {}_nm_a(t) = \frac{\int_t^{t+n}\int_a^{a+n}l(a,t)\mu(a,t)da\:dt }{\int_t^{t+n}\int_a^{a+n}l(a,t)da\:dt}

Let's not write out the double-integral for all examples below, but Dismod-AT does
perform its integration over both age and time. Instead, write
the following short-hand,

.. math::

    {}_nm_a(t) = \frac{\mbox{death events}}{\mbox{Susceptible + With-Condition life-years}}.

for the integral.

Similarly, the population susceptible incidence rate is

.. math::

    \frac{\mbox{incidence events}}{\mbox{Susceptible life-years}}

The population remission rate has the same problem as the incidence,
in that it can be counted as a percentage of those with condition
who remit or a percentage of the population that remits.
If we consider the remission hazard rate, which is the former,
then it is

.. math::

    \frac{\mbox{remission events}}{\mbox{With-Condition life-years}}

.. note::

    We could define a t-remission as

    .. math::

        \frac{\mbox{remission events}}{\mbox{Susceptible + With-Condition life-years}}.

    but we don't. Is that because all remission is of one type or another? Which type?

The population excess mortality rate is

.. math::

    \frac{\mbox{excess death events}}{\mbox{With-Condition life-years}}.

Other-cause mortality is just like mortality, but only
for susceptibles,

.. math::

    \frac{\mbox{death events just from susceptible}}{\mbox{Susceptible life-years}}.

The population rate for ``mtall`` and ``mtspecific`` both use :math:`S(x)+C(x)` as their
weight. The same is true of standardized mortality ratio and relative risk.


.. note::

    Dismod-AT expects the user to provide weight functions. The GBD provides
    weight functions, which should correspond to :math:`S(x)+C(x)`. These should
    also be close enough for :math:`S(x)`. It would make sense to create
    and refine the weight corresponding to :math:`C(x)` as we solve
    down the location hierarchy.


Crude Population Rates
----------------------

Dismod-AT works with life table rates, not crude rates. A crude rate
is the number of deaths divided by the number of people
exposed to that event. If :math:`k(t)` is the birth rate over
time, then a crude mortality rate is

.. math::

    {}_nM_x = \frac{\int_x^{x+n}k(t-a)l(a)\mu(a)da }{\int_x^{x+n}k(t-a)l(a)da}

The life table rate adjusts the crude rate
to remove the effect of varying birth rates. In Dismod-AT,
the birth rate is normalized to a rate of 1 for all populations.
In demographic textbooks, :math:`{}_nm_x` is called the lifetable
mortality rate, and :math:`{}_nM_x` is called the crude mortality rate.

.. note::

    The bundles aggregate measurements from many sources.
    Do they use crude population rates or lifetable population rates?

This matters when there is a birth pulse that skews data towards
younger or older sides of an age interval. Dismod-AT assumes that
the average over an age interval is determined by the lifetable
person-years lived.
