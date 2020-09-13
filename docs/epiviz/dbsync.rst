.. _db-autosync:

Auto Sync of DB tables so we can run MR and AT in parallel
===========================================================

Scripts run daily on Jenkins to sync 11 tables from ``modeling-epi-db`` to ``epiat-db-p01``
and ``epi-db-d01``.  The 11 tables are:

* ``epi.modelable_entity``
* ``epi.modelable_entity_cause``
* ``epi.modelable_entity_metadata``
* ``epi.modelable_entity_metadata_type``
* ``epi.modelable_entity_note``
* ``epi.modelable_entity_rei``
* ``epi.modelable_entity_set``
* ``epi.modelable_entity_set_version``
* ``epi.modelable_entity_set_version_active``
* ``epi.modelable_entity_type``
* ``epi.study_covariate``
  
If there are no changes, nothing is replicated.


The sync can be run manually, if desired, by: 

#. logging into Jenkins: https://centralcompdb-jenkins.ihme.washington.edu/

#. clicking on the job: **Epi_Refresh_From_modeling-epi-db_modelable_entity_study_covariate**

#. clicking on: **Build Now**


To see the result of the run: 

#. click on the build link

#. click on Console Output


The scripts are located here:
https://stash.ihme.washington.edu/projects/IHMEDB/repos/epi/browse/shell_scripts

and they are: 

* ``epi-pt-table-epi-db-d01-run.sh``	DD-1091, DD-1095
* ``epi-pt-table-epiat-run.sh``	        DD-1091, DD-1095	
* ``epi-pt-table-sync-list-tables.sh``	DD-1091, DD-1095
