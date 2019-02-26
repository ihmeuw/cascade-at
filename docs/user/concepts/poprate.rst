
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

