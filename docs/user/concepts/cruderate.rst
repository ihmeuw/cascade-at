.. _crude-population-rates:

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
