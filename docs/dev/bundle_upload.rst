.. _bundle-upload-full-datasets:

Bundle Upload for Full Datasets
===============================

The goal is to upload a new synthetic dataset for the whole world.
These synthetic datasets are constructed so that we have a known
true answer (called an oracle) for figuring out how to fit.

Most people are uploading a bunch of data and the maybe going back and
adjusting a few rows for typos or whatever.
But we're re-uploading whole data sets over and over. It's a different pattern.

It's something we can automate. We just need to know what the ``seq`` are
for the existing bundle. There's a function for that.
Then you can either issue a deletion request separately or append the
deletions to the bundle being uploaded or align those ``seq`` with
some/all the rows in the new data set so it overwrites.
The last option is probably the fastest at execution time but
the first option is simplest to write. And is what I've been doing manually


 1. Run the data base to bundle script in the cascade repo giving it
    the input path, output path and bundle id::

        scripts/db_to_bundle.py new_base_case.db new_base_case.xlsx 4319

 2. Then delete the existing bundle data using the modify bundle script
    on the cluster::

        /ihme/code/dismod_at/bin/modify_bundle --bundle-id 4319 \
            /homes/alecwd/delete.xlsx

    That assumes that your new bundles have the same number of
    rows as the old ones. If they don't you'll need to make
    a new deletion spreadsheet.

 3. Then upload the new data::

        /ihme/code/dismod_at/bin/modify_bundle --bundle-id 4319 \
            /path/to/new_base_case.xlsx

    Remember to update the bundle id to match the one you are working on.

The deletion bit in the middle is awkward but it's a side effect
of how the GBD process thinks about this kind of data.
If we end up doing a lot of this we can automate
it but so far it hasn't been worth it.
