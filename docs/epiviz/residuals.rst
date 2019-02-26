.. _epiviz-residuals:

Result Residuals
================

There are prior residuals and data residuals.

These indicate how far each prior was from the value used in the fit.
There is a prior at each age and time point in the fit and on the
standard deviation multipliers. There are three times as many priors
as there are fit points. There are priors for every estimation run,
including the simulations. The simulation residuals are often thrown out.

These are one value per data point to indicate how far the data, as
predicted by the fit, is from the actual measurement data. There are data
residuals for every estimation run, so each data point will have five
residuals for the five levels where it is used.
