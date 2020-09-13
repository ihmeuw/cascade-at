.. _file-locations:

File Locations on the Cluster
=============================

 *  ``/ihme/epi/dismod_at``

    *  ``share/local_odbc.ini`` - This is an ODBC file to use when running on
       the Fair cluster. This is used in ``dismodel_main.py`` while
       their password technique involves using the unavailable J drive.

    *  ``bin/`` - Contains tools for modelers to manipulate bundles on
       the debug database. Subdirectories of this contain hacked copies
       of the uploader.

    *  ``log/`` - The MATHLOG, by model version id, is in this directory.

    *  ``runscripts/`` - The shell scripts run by EpiViz-AT are here.
       They are copied from the scripts directory in the Github.

 *  ``/ihme/epi/at_cascade``

    *  ``logs/`` - This log is from the shell file that runs qsub submission.

    *  ``prod/`` and ``dev/`` - Data is in here.

 *  ``/ihme/temp/sgeoutput/<user>/dismod/`` Logs out of qsub stdout and stderr.
