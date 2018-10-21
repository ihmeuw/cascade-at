.. _overview-of-epiviz:

Overview of EpiViz Runner
=========================

.. _epiviz-overview-introduction:

Introduction
------------
EpiViz-AT is a web page on http://epimodeling-web-d01.ihme.washington.edu/at
or http://epimodeling-web-p01.ihme.washington.edu/at. After you fill out
that web page and hit submit, the web server runs a program on the cluster.
We call that program ``epiviz_runner``. It installed on the IHME cluster,
and the code itself is stored at https://github.com/ihmeuw/cascade.
The ``epiviz_runner``, itself, is at
https://github.com/ihmeuw/cascade/blob/develop/src/cascade/executor/epiviz_runner.py.

This page describes what that code does, in order.

.. _epiviz-overview-outline:

Outline
-------

.. autofunction:: cascade.executor.epiviz_runner.main


.. _build-model-from-epiviz-settings:

Building a Dismod-AT Model from EpiViz Settings
-----------------------------------------------

There are complicated rules for how we build Dismod-AT models.
There are multiple data sources, and many decisions to make
about how to enter that data. Here is how we separate them
in EpiViz Runner.

.. autofunction:: cascade.executor.epiviz_runner.model_context_from_settings


.. _convert-bundle-to-measurement-data:


Steps to Convert Bundle to Measurement Data for Dismod-AT
---------------------------------------------------------

At the start of this process, the measurement bundle is on
the Epi database. By the end of this process, there is
a table of measurements in the Dismod-AT input file.

 1. Retrieve the measurement bundle by ``bundle_id`` from the
    epi database.

 2. Convert each record in the measurement bundle from its
    GBD measure ID to a Dismod measure, according to the
    `bundle id map <https://github.com/ihmeuw/cascade/blob/develop/src/cascade/input_data/configuration/id_map.py>`_.
    If the incoming data uses incidence, instead of S-incidence
    or T-incidence, reject it.

 3. Convert sex id from 1, 2, 3, 4 to male, female, both, or unspecified.

 4. Find the list of study covariates associated with this bundle,
    both by covariate id and by name.

 5. If data is in demographic notation, then the bundle end times
    will be wrong, but the code prints a message to say whether it adds
    a year to the end of an age region or time region.

 6. It assigns a set of weights to the measurements. These are
    meant to give relative importance according to changes in
    population size. These are all 1 if the weight method is "constant."

 7. All observations with standard error at zero are removed,
    unless they are relative risks.
