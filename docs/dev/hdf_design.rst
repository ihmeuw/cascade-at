.. _hdf-design:

HDF Design
==========

This is how to format an HDF5 storage file.
The main tool is `h5py <http://docs.h5py.org/en/stable/index.html>`_,
whose documentation is sufficient. For more background, read
the `HDF5 Documentatiion <https://support.hdfgroup.org/HDF5/doc/index.html>`_.

This is a file format both for single runs of Dismod-AT through the Cascade
and for full global runs. It have an atomic store-this-one-fit and a larger
set of rules around how to store multiple fits for a global Cascade.

For the global Cascade, separate observation from estimation from sampling
because the structure in the HDF file is both for finding data and for
managing data, and observations (input data) and samples can be deleted
separately. Observation data can also be shared among estimations.
In order to manage these three directories separately, it is much simpler
if there are no links from one to the other. HDF doesn't store data in the
group hierarchy you assign. It presents data that way. Data isn't deleted
until all links to that data are gone.

 * ``observation/`` These are inputs that look like Pandas dataframes.
    * ``measurement/`` This is data from surveys and hospital data.
    * ``asdr/`` Age-specific death rates.
    * ``csmr/`` Cause-specific mortality rates.
 * ``estimation/`` Contains the bulk of the global work.
    * ``{location_id}/`` Could be separated by location, or other designation.
       * Cascade info
       * Main fit
       * Output from simulations.
    * ``weights/`` These are shared down the hierarchy.
    * ``covariates/`` These will also be shared down the hierarchy.
    * ``locations/``
 * ``sampling/`` A mirror structure contains runs for sampling around fit results.
    * ``{location_id}/{sample_id}/`` A model is stored in here.

There are two levels at which to think about a single Dismod-AT fit.
One is a single run of the Cascade main, where it's a recipe for running
init, fit, simulate. The other is a single call to dismod fit. Let's tackle
the single fit-and-predict first.

 * Input measurements
 * Desired output measurements (avgint)
 * Stats model
   * Smooth grid of priors
   * starting var
   * parent and child locations.
   * scale var
   * weights
 * Output
   * fit var
   * residuals
   * estimated data out
 * Tracing data on the run
 * Provenance on the run
 * Logs from Dismod-AT

We run a fit-and-predict multiple times, and we track not only the Dismod-AT
run but also how the Cascade decided to do sets of runs.

 * Inputs to Cascade
   * JSON settings
   * posteriors
   * weights
   * covariate multipliers, if they were set higher in hierarchy
   * measurements, asdr, csmr
 * main fit
 * fit with uncertainty (from fit_var on all samples)
 * estimated data with uncertainty (from predict)
 * 10-1000 sample fits
 * Cascade log
 * Cascade tracing
 * Cascade provenance
