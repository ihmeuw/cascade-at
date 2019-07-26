.. _dmcascade-command-line:


Command-line EpiViz-AT
----------------------

Run the cascade against the latest model version associated with an meid::

    export PATH=$PATH:/ihme/code/dismod_at/bin
    dismodel --meid 1989

This will create subdirectories of the current directory that look like
``./meid/mvid/1/location_id``. The 1 is a directory that splits locations
into sets of 100, so numbers of files within a directory don't get out of hand.
Call from a JSON settings file::

    dismodel --settings-file 1989.json

Run from bundle data on disk rather than loading it from the database::

    dmcascade --settings-file 1989.json --bundle-file inputs.csv --bundle-study-covariates-file input_covs.csv

The bundle file must have the following columns: seq (id used to align with the study covariates data), measure, mean, sex_id, standard_error, hold_out, age_lower, age_upper, time_lower, time_upper, location_id

The study covariates file must have the following columns: seq (must match the id in the bundle file), study_covariate_id


**Dismodel**

.. program:: dismodel

.. option:: -h, --help

    Prints help

.. option:: -v, --verbose

    Increases logging to stderr

.. option:: -q, --quiet

    Decreases logging

.. option:: --logmod LOGMOD

    Given a module named LOGMOD, sets its logging level using ``modlevel``

.. option:: --modlevel LOGLEVEL

    LOGLEVEL is one of debug, warning, info, error, for the modules
    listed by ``logmod``.

.. option:: --mvid MVID

    MVID is a model version ID to retrieve from databases.

.. option:: --meid MODELABLE_ENTITY_ID

    Read the latest settings for this modelable entity ID. This will get
    a specific model version ID.

.. option:: --settings-file SETTINGS

    SETTINGS is a JSON-formatted file that comes from downloading
    settings from the EpiViz-AT UI or by using the ``dmgetsettings`` tool.

.. option:: --base-directory DIRECTORY

    File writing and reading is relative to this directory.

.. option:: --no-upload

    Do the calculation, make the files, but don't save it to the databases
    that infrastructure uses.

.. option:: --db-only

    Download settings and data, build the Dismod-AT db file for the initial
    fit, but don't do the fit.

.. option:: --infrastructure

    If this flag is used, then use the modelable entity id and model version
    id to create a nested directory structure under the base directory.

.. option:: --skip-cache

    Download all data directly from the databases, instead of going to tier 3,
    which is a version stored in order to ensure repeatable runs. This flag
    also turns off creation of a tier 3 copy of data.

.. option:: --num-processes NUM_PROCESSES

    NUM_PROCESSES is an integer number of subprocesses to use when calling
    Dismod-AT fit simulate, which generates draws. The number of processes
    should fit within the computer's memory for the given model.

.. option:: --pdb

    If the program encounters an error then it will drop into a debugger
    if this flag is given.
