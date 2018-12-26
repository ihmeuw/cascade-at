Examples Calling Dmcascade
==========================

Retrieve a Settings File from EpiViz-AT
---------------------------------------

If you want to run and modify a model configured in EpiViz-AT already,
try downloading the settings file for it. Then edit the settings
file and run ``dmcascade`` on it, as described below.

These commands are all in ``/ihme/code/dismod_at/bin``.
First download the file by its model version ID::

    export PATH=$PATH:/ihme/code/dismod_at/bin
    dmgetsettings --mvid 266870 anx.json

That will create a file called ``anx.json`` in the JSON format.
It's long, but it looks, in part, like this::

    "rate": [
        {
            "default": {
                "value": {
                    "density": "gaussian",
                    "min": 1e-08,
                    "mean": 0.002,
                    "max": 0.01,
                    "std": 0.01
                },
    ...

Entries in the settings file are interpreted by code in
https://github.com/ihmeuw/cascade/blob/develop/src/cascade/input_data/configuration/form.py, which can help understand what to set.

There is a second way to get the settings, which is to get them
from the web browser while you use EpiViz-AT.
First run EpiViz-AT.
Then go to the Developer Console.
This is a menu option in the web browser.
Then type ``printSettingsJson()``. Copy to a file what that command prints.


Command Line
------------
Run the cascade against the latest model version associated with an meid::

    export PATH=$PATH:/ihme/code/dismod_at/bin
    dmcascade 1989.db --meid 1989

Call from a JSON settings file::

    dmcascade 1989.db --settings-file 1989.json

Run from bundle data on disk rather than loading it from the database::

    dmcascade 1989.db --settings-file 1989.json --bundle-file inputs.csv --bundle-study-covariates-file input_covs.csv

The bundle file must have the following columns: seq (id used to align with the study covariates data), measure, mean, sex_id, standard_error, hold_out, age_lower, age_upper, time_lower, time_upper, location_id 

The study covariates file must have the following columns: seq (must match the id in the bundle file), study_covariate_id
