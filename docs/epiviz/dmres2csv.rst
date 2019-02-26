DMRES2CSV Tool
==============

This is a script that retrieves model residuals for Dismod AT models
and writes two csv files - one for residuals from the ``fit_var`` table, and one for 
residuals from the ``fit_data_subset`` table.  

The input required is the name of a Dismod sqlite db file.  Optionally, an
output directory can be specified for the csv files, otherwise, they will be
written to the current directory.  A ``model_version_id`` can be provided to help 
identify the models in the name of the csv files.

The output csv files have the names: ``resids_fv_<mvid>.csv`` and ``resids_fds_<mvid>.csv``, 
where fv stands for ``fit_var``, fds for ``fit_data_subset`` and <mvid> stands for the ``model_version_id`` 
provided.  

The columns in the output files are the same as those in the ``fit_var`` 
and ``fit_data_subset`` tables.

``fit_var`` columns:

 * fit_var_id
 * variable_value
 * residual_value
 * residual_dage
 * residual_dtime
 * lagrange_value
 * lagrange_dage
 * lagrange_dtime

``fit_data_subset`` columns:

 * fit_data_subset_id
 * avg_integrand
 * weighted_residual


To run the script on the cluster:

1. Activate the current cascade environment::

    cluster> source /ihme/code/dismod_at/env/current/bin/activate

2. Run the script and supply values for <x>, <y>, <d>::

    cluster> dmres2csv --dm-file=<x> --mvid=<y> --output-dir=<d> 


An example call to ``dmres2csv`` is::

    dmres2csv --dm-file=diabetes.db --mvid=102680 --output-dir=/ihme/code/someusername/somedir 

Two csv files will be written to ``/ihme/code/someusername/somedir``::

    resids_fv_102680.csv and resids_fds_102680.csv
