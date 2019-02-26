.. _epiviz-draws:

Result Draws
============

There are two kinds of draws, integrand draws and fit draws.

Results of calculations are in "draw files." These draw files
currently include integrands on the GBD age and time intervals and for each
draw, which may number a thousand at the most-detailed level.
7 integrand types (S-incidence, T-incidence,
CSMR, etc) x 23 age groups x (2019 - 1990) x (draw count).

The output of the main fit and of the draws, which determine uncertainty,
is an interpolation over age and time. This is different from the draws
described below. It's specified by rate values on an age and time grid.
If there are seven years and 23 age points, there are 161 values.
Fits also include "standard deviation multipliers" which are hyper-priors
on the standard deviation of the priors. There are up to three per
underlying rate in the problem. There can be five rates, and each rate
generally has about 30 age points. A rate can have five year points or
a point for every year, which is about 68 points. So this number is
between 3 rates x 5 years x 30 points = 450 and 5 rates x 68 years
x 30 points = 10200 points. This is per-run of Dismod-AT, so there would
be 31 of these if we save 30 simulations and 1 main fit.
