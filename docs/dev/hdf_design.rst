.. _hdf-design:

HDF Design
==========

Why Make an HDF File Format?
----------------------------

The Cascade is a distributed computation that can take more than a week.
It has three storage needs:

 1.  Inputs come from databases of measurement data, mortality rates, and
     covariates. Because inputs can change, they are cached in a database
     called "tier 3." This document proposes to store the same data
     in a file because that protects from database inaccessibility and
     would permit computation in the cloud.

 2.  Estimation runs at different levels of the hierarchy exchange
     results in order to inform the next level down the hierarchy.
     This has to be stored somewhere, and HDF is the default format.

 3.  The final results can be large and should be efficient for reading,
     more than for writing. Here, we can make a significant improvement
     on how easily draws can be read because we can control the order
     in which data is stored, across draws, ages, times, and measures.

In sum, this provides a single file to record a whole Cascade while,
at the same time, it permits data management within that file. For instance,
it is possible to delete all draws from all of the files without
disturbing descriptions of input data or models.


Structure
---------

The main tool is the library `h5py <http://docs.h5py.org/en/stable/index.html>`_,
whose documentation is good. For more background, read
the `HDF5 Documentation <https://support.hdfgroup.org/HDF5/doc/index.html>`_.

This is a file format both for single runs of Dismod-AT through the Cascade
and for full global runs. It have an atomic store-this-one-fit and a larger
set of rules around how to store multiple fits for a global Cascade.

This document plans how to put everything about a global Cascade run
into one file, but that doesn't mean all of the data in that same file
has the same lifetime, the same compression, or the same clients reading
and writing. The structure of the HDF file will reflect the different
kinds of data management. We put data that is managed similarly into
the same top-level HDF Group. Those three groups will be input data,
estimation models, and output data. While we use the three top-level
groups for management, HDF's implementation
doesn't store data in the group hierarchy you assign. It presents
data that way. The way HDF works,
data isn't deleted until all links to that data are gone.

 * ``input/`` These are inputs that look like Pandas dataframes.
    * ``measurement/`` This is data from surveys and hospital data.
    * ``asdr/`` Age-specific death rates.
    * ``csmr/`` Cause-specific mortality rates.
    * ``covariates/`` These will also be shared down the hierarchy.
    * ``locations/``
    * ``epiviz_settings`` Specification from the user interface.
 * ``estimation/`` Contains the bulk of the global work.
    * ``{location_id}/`` Could be separated by location, or other designation.
       * Cascade info
       * Main fit
       * Output from simulations.
       * ``weights/`` There are three weights: susceptible, with-condition, and total.
       * scaling variables if used.
       * initial guess, if used.
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
