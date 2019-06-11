.. _metrics-for-timing:


Metrics for Timing Dismod-AT
============================

There are three main sources of numbers to use for testing.

 1. Measurements of resource usage, from
    https://github.com/ihmeuw/rocketsonde or from ``/usr/bin/time -v``.

 2. Metrics on input data and parameters, as described below and
    implemented in ``cascade.dismod.metrics``.

 3. Observations of what the program did, retrospectively, which
    means counting the number of iterations it used, for instance.
    These are done in ``cascade.dismod.process_behavior``.

We need to characterize the memory usage and runtime of Dismod-AT.
Here are factors that may contribute and why they may contribute.

 *  total number of data points, because each data point is compared
    separately against the fit.

    -  Number of points for which ``INTEGRAND_COHORT_COST`` is true. These
       are the points that Greg calls not primary, but relative risk is
       considered as not being costly, according to Dismod-AT's documentation.

    -  Number of points with age extent or time extent.

    I would store the four categories separately, as defined by the
    two conditions above. We can add them later as needed.

 *  number of age steps used for integration. Not worried about the exact
    age steps, just how many there are.

 *  total extent in age and total extent in time.

 *  number of smoothing grids, in total. This counts random effects
    as a separate smoothing grid per child.

 *  number of random effect grid points, so a random effect with one
    point is one, but with twenty points counts as twenty. Add those
    up. It's probably the zero, or not zero, that counts most.

 *  number of model variables, because this is the count of unknowns,
    although it includes constant values.

 *  Options to Dismod-AT are single values that can greatly change
    the memory and time characteristics. These are worth examining:

    -  ``zero_sum_random``
    -  ``derivative_test_fixed``
    -  ``derivative_test_random``
    -  ``max_num_iter_fixed```
    -  ``max_num_iter_random``
    -  ``tolerance_fixed``
    -  ``tolerance_random``
    -  ``quasi_fixed``
    -  ``bound_frac_fixed``
    -  ``limited_memory_max_fixed``
    -  ``bound_random``

All of this can be collected from the session and model just before running
Dismod-AT.

Aaron suggested a nonparametric regression technique to help with fitting
curves for this data: https://cran.r-project.org/web/packages/acepack/index.html.
