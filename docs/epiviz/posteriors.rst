.. _epiviz-posteriors:

Posteriors as Data
==================

.. warning::

    This is not implemented.


One way to initialize a child estimation is to generate predicted integrands
from the parent estimation and include them as data in the child. There are
two differences between these integrands and the draws.
While the posteriors-as-data come from simulated fits, individual simulation
values aren't required. The mean and standard error are saved for child estimation.
The other difference is that these integrands can be point values, meaning
they have no age or time extent. It is also appropriate to sample
these points not at the midpoint of age intervals but at the boundaries
of age intervals.

The shape of this data should be as follows.

 *  ``location_id``-  A child location ID, covering all children. The parent
    location is not included for fits with random effects. For a fit
    with fixed effects, this location is the parent location.

 *  ``integrand`` - String name of the integrand. All 13 of the integrands
    will be represented here.

 *  ``age_lower`` and ``age_upper`` - These two values will be equal. They
    will be at all boundaries of the 23 GBD age groups, which are 23 values
    because the last GBD age group is a half-open interval.

 *  ``time_lower`` and ``time_upper`` - These will be equal and at
    mid-year, so 2000.5, 2005.5.

 *  ``sex_id`` - Will be 1 for a Male drill, 2 for a Female drill.
    (I'm a little unclear on what can happen here for a drill for both sexes.)

 *  ``mean`` - The mean is taken from the a-posteriori mean, not the
    mean of the draws.

 *  ``std`` - The standard deviation, taken from MLE of each location,
    age, time, and sex in the draws. The MLE is taken by setting
    the mean to the a-posteriori mean and optimizing, given that value
    for the mean and a Gaussian distribution.

 *  The distribution isn't specified and is assumed to be Gaussian.

If the fit uses fixed effects, and not random effects, then the data
will be the parent location's data.
