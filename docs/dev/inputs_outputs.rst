.. _inputs-and-outputs:

Inputs and Outputs
==================

Inputs
------

Tier 3 Data
  The input data is *partially* recorded in the tier 3 storage, which contains
  the bundle, covariates, age-specific death rate (ASDR), and cause-specific mortality
  rate (CSMR). Not in the tier 3 data are location sets, age groups, year ids, or
  sex id definitions. Derived or imputed data, such as excess mortality rate (EMR)
  isn't stored there, either.

   * Bundle - As many data points as we have for the world.
   * Covariates - Number of covariates times number of nonzero entries for this bundle.
   * ASDR - Yearly from 1950-present on 23 age groups = 1564 values.
   * CSMR - Yearly from 1950-present on 23 age groups = 1564 values.

EpiViz-AT Settings
  There is also a record of what the Cascade plans to calculate,
  contained in the EpiViz-AT settings. These record, for instance, that
  excess mortality rate was imputed from prevalence data or that cause-specific
  mortality was included from a different cause than the main one.
  There are usually under 200 key-value entries but can be as many as
  three times the age-time grid points, around 2000 key-value entries.
  This will happen when a modeler uses
  code to create a settings file and will be rarer.

Intermediate Files
------------------

DB Files
  At each location, the Cascade writes a "db file,"
  which is a file format for Dismod-AT input and output. Those db files
  are a record of exactly what data Dismod-AT sees and produces and are useful
  for debugging. The db files contain all outputs except the Cascade logs.
  The db files are in an sqlite3 format defined by Dismod-AT.
  The size of a DB files is determined by the size of the data and the
  number of age-time grid points. Assume 2000 grid points, and the data
  can be quite large for global, super-region, and region solutions.

Posterior Fits
  Each estimation run, meaning one fit of Dismod-AT for a parent location
  and its children, generates output that may be used to choose priors
  for the next level down in the Cascade. It is fits that are passed down,
  so these are the underlying rates (incidence, remission, excess mortality,
  other-cause mortality, initial prevalence) on every point of the age-time
  grids, so there are about 23 ages x 5 times x 4 grids = 460 values.

Posterior Integrands
  In addition to passing down posteriors as priors, we can use data as priors.
  These are integrands from the output of a level above being used to adjust
  a fit below. Integrands used in visualization have age-time extent, so they
  apply to 45-50 year-olds from 1990-1991. The integrands for priors will be
  a separate set on point-ages and point-times, so 50 year-olds at 1991.
  They will be on GBD age-time grids (24 ages, 5 or 68 years)
  and will include at least prevalence, maybe the primary rates, which are
  the five that correspond to the underlying rates.

Outputs
-------

Cascade Logs
  The logs record what the Cascade actually chose to do when it saw
  inputs and settings, although they are difficult to parse. The reason
  to parse them is to see how interactions between what the user requested
  through the EpiViz-AT user interface and what the databases provide resulted
  in a final model. There is one log per estimation run.

Dismod-AT Logs
  These get embedded in the Cascade logs, but it is worth mentioning that
  the log-level used for Dismod-AT can make the logs get very large,
  because Dismod-AT can log every step of its nonlinear solver. Dismod-AT
  logs have two levels, a high-level report from Dismod-AT itself, and
  a low-level report from the Ipopt optimizer that converges within each
  of the Dismod-AT time steps. We need the Dismod-AT convergence data more
  than data on iterations by Ipopt.

Fits
  The output of the main fit and of the draws, which determine uncertainty,
  is an interpolation over age and time. This is different from the draws
  described below. It's specified by rate values on an age and time grid.
  If there are seven years and 23 age points, there are 161 values.
  Fits also include "standard deviation multipliers" which are hyper-priors
  on the standard deviation of the priors. There are up to three per
  underlying rate in the problem. There can be five rates, and each rate
  generally has about 30 age points. A rate's smooth grid tends to have
  either one time point, about five time points, or
  a point for every year, which is about 68 points. So this number is
  between 3 rates x 5 years x 30 points = 450 and 5 rates x 68 years
  x 30 points = 10200 points. This is per-run of Dismod-AT, so there would
  be 31 of these if we save 30 simulations and 1 main fit.

Prior Residuals
  These indicate how far each prior was from the value used in the fit.
  There is a prior at each age and time point in the fit and on the
  standard deviation multipliers. There are three times as many priors
  as there are fit points. There are priors for every estimation run,
  including the simulations. The simulation residuals are often thrown out.

Data Residuals
  These are one value per data point to indicate how far the data, as
  predicted by the fit, is from the actual measurement data. There are data
  residuals for every estimation run, so each data point will have residuals
  for every level of the location hierarchy where it is used, so that's about
  five.

Draw Files
  Finally, results of calculations are in "draw files." These draw files
  currently include integrands on the GBD age and time intervals and for each
  draw, which may number a thousand at the most-detailed level.
  7 integrand types (S-incidence, T-incidence,
  CSMR, etc) x 23 age groups x (2019 - 1990) x (draw count).


Number of Estimations
---------------------

Each location-specific estimation of Dismod-AT generates more data, so let's
count them.

At one parent location, the Cascade estimates rates for all child locations.

 *  **Fit** Runs a Dismod-AT fit to find the most likely estimate.
 *  **Simulation** Runs :math:`s=30` more fits to determine error in that most likely estimate.
 *  **Predict** Runs predict on each of those fits in order to get error for integrands.

Given a location hierarchy with six levels, the most-detailed, or leaf, level
doesn't need a separate run. If the levels are global, super-region, region,
national, and sub-national, then Dismod-AT runs on global, super-region, region,
and national, which is four levels. At each of those levels, there will be
a single fit to find the maximum and then 30 simulations, all 30 of which
can run simultaneously.

