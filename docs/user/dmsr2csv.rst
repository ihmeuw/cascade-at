DMSR2CSV Tool
=============

This is a script that retrieves model results for Dismod AT and ODE models
and writes it as a csv file for each model type.  The data is retrieved from
the ``epi.model_estimate_fit`` table.  The database used for AT is ``dismod-at-dev``
and for ODE is ``epi``.

The input required is a ``model_version_id`` for each model type.  Optionally, an
output directory can be specified for the csv files, otherwise, they will be
written to the current directory.  If an invalid ``model_version_id`` is provided
or if no ``model_version_id`` is provided, an empty dataframe will be retrieved and 
a csv file with no rows of data will be written.  

The output is a csv file for an AT model and a csv file for an ODE model.
These files have the names: ``at_<mvid>.csv`` and ``ode_<mvid>.csv``, where <mvid> 
stands for the model_version_id provided for each model type.  

The columns in the output files are the same as those in the ``epi.model_estimate_fit`` 
table:

 * model_version_id
 * year_id
 * location_id
 * sex_id
 * age_group_id
 * measure_id
 * mean
 * upper
 * lower

To run the script on the cluster:

1. Activate the current cascade environment::

    cluster> source /ihme/code/dismod_at/env/current/bin/activate

2. Run the script and supply values for <x>, <y>, and <d>:: 

    cluster> dmsr2csv --at-mvid=<x> --ode-mvid=<y> --output-dir=<d>     


An example call to ``dmsr2csv`` is::

    dmsr2csv --at-mvid=265844 --ode-mvid=102680 --output-dir=/ihme/code/someusername/somedir

Two csv files will be written to ``/ihme/code/someusername/somedir``::

    at_265844.csv and ode_102680.csv




