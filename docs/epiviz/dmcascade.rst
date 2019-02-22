.. _dmcascade:


Command-line EpiViz-AT
----------------------

Run the cascade against the latest model version associated with an meid::

    export PATH=$PATH:/ihme/code/dismod_at/bin
    dmcascade 1989.db --meid 1989

Call from a JSON settings file::

    dmcascade 1989.db --settings-file 1989.json

Run from bundle data on disk rather than loading it from the database::

    dmcascade 1989.db --settings-file 1989.json --bundle-file inputs.csv --bundle-study-covariates-file input_covs.csv

The bundle file must have the following columns: seq (id used to align with the study covariates data), measure, mean, sex_id, standard_error, hold_out, age_lower, age_upper, time_lower, time_upper, location_id

The study covariates file must have the following columns: seq (must match the id in the bundle file), study_covariate_id
