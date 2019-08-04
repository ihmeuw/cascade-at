.. _timing-dismod-at:

Timing Dismod-AT
================

.. _code-for-timing:

Code for Timing
---------------

We think that most of the runtime for each Cascade job will be running
Dismod-AT and that the Python part of the code will be fairly
predictable and not memory intensive. Therefore, we time Dismod-AT itself
in order to characterize run times and memory usage.

Dismod-AT's run times and memory usage can vary a lot, depending on parameters.
There isn't a single number, such as the number of input data records,
that is the most important for estimating run time and memory usage.
Therefore, we will run it a lot of times and use statistical approaches
in order to make a guess about run time and memory usage.

The strategy is to run Dismod-AT many times in different configurations.
We use Dismod-AT's predict function in order to create fake data,
so that we know the answer before we ask it to fit.
This is an experiment, and the experimental design used here is
called fractional factorial because it defines a baseline set of
important parameters and then variations on all those parameters. Then
it walks one parameter at a time through its variations. Then it walks
two parameters at a time to see interaction. Then it walks three,
and so on, up to a combination you decide. Doing two at a time
requires 3850 runs.

The code for timing is in the ``examples/`` directory. The
relevant files are

 * ``main_success_scenario.py`` - This is a main that runs Dismod-AT
   any one of thousands of ways. Each way to run is indexed by a single
   number for replay. *This file doesn't constrain omega, and it should,
   because constraining omega changes the runtime behavior of Dismod-AT
   significantly.*

 * ``main_success.sh`` - A shell file for Grid Engine to run the
   main success scenario many times.

 * ``main_success_gather.py`` - The JSON files created by Python aren't
   readable by R. Probably Python's fault. This reads those files and
   makes a single CSV out of them.

 * ``gathertiming.R`` - This analyzes the data from the main success
   scenario. This uses R's acepack in order to see which parameters
   matter most.

 * ``add_integrands.py`` - This is a main that goes through the data
   in order to re-analyze the db files that are created.

This would be an excellent unit test, except that the
main success scenario *often doesn't run.* Dismod-AT is just really
hard to set up, even if you create data where you know the right answer.


.. _metrics-for-timing:

Metrics
-------

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

These metrics end up in two places:

 * cascade.core.subprocess_utils - Runs ``/usr/bin/time``.
 * cascade.dismod.metrics - Collects db file data.
