.. _inputs:

Data Inputs for Cascade-AT
==========================

Wrangling the inputs for a Cascade-AT model is a very important first step.
All of the inputs at this time come from the IHME epi databases. In the future
we'd like to create input data classes that don't depend on the epi databases.

:ref:`input-components` documents the inputs that are pulled
for a model run. :ref:`measurement-inputs` documents how each of those
inputs works together to create one large object that stores all of the
input data for a model run (including each of the input components).
:ref:`input-demographics` describes the demographic and location
inputs that need to be set for a model.

.. toctree::
   :maxdepth: 3

   input-demographics
   input-components
   measurement-inputs
