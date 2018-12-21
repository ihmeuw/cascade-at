DMSR2CSV Tool
=============

This is a script that retrieves model results for Dismod AT and ODE models
and writes it as a csv file for each model type.  The data is retrieved from either
the ``epi.model_estimate_fit`` or ``epi.model_estimate_final`` table from one of these
databases, ``dismod-at-dev``, ``dismod-at-prod``, ``epi``, or ``epi-test``.

The input required is a ``model_version_id`` for each model type.  Optionally, an
output directory can be specified for the csv files, otherwise, they will be
written to the current directory.  If an invalid ``model_version_id`` is provided
or if no ``model_version_id`` is provided, this will be reported and no data will be
written.

The database to use, and the table within that database, can be supplied for each 
``model_version_id``.  If the database and table are not both supplied, the possible 
combinations from the database and table lists above are checked for a given ``model_version_id``.  
If data is found in more than one of the database+table combinations for a ``model_version_id``, 
the locations found are reported, and no data is written for that ``model_version_id``.  If the 
database and table are provided in a new run, the data will be retrieved and written.

The output is a csv file for an AT model and a csv file for an ODE model.
These files have the names: ``at_<mvid>.csv`` and ``ode_<mvid>.csv``, where <mvid> 
stands for the ``model_version_id`` provided for each model type.  

The columns in the output files are the same as those in the ``epi.model_estimate_fit`` 
and ``epi.model_estimate_final`` tables:

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

2. Run the script and supply values for <x>, <y>, <d>, <xdb>, <xt>, <ydb>, <yt>:: 

    cluster> dmsr2csv --at-mvid=<x> --ode-mvid=<y> --output-dir=<d> --at-db=<xdb> --at-table=<xt> --ode-db=<ydb> --ode-table=<yt>     


The possible values for <xdb> or <ydb> are: "dismod-at-dev", "dismod-at-prod", "epi-test", or "epi"

and 

the possible values for <xt> or <yt> are: "fit" or "final"

Both the database and table should be specified, if using these arguments (not just one of the two).  



An example call to ``dmsr2csv`` is::

    dmsr2csv --at-mvid=265844 --ode-mvid=102680 --output-dir=/ihme/code/someusername/somedir --at-db="dismod-at-dev" --at-table="fit" --ode-db="epi" --ode-table="fit"

Two csv files will be written to ``/ihme/code/someusername/somedir``::

    at_265844.csv and ode_102680.csv




