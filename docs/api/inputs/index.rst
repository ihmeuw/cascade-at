.. _inputs:

Data Inputs for Cascade-AT
==========================

Wrangling the inputs for a Cascade-AT model is a very important first step.
All of the inputs at this time come from the IHME epi databases. In the future
we'd like to create input data classes that don't depend on the epi databases.

:ref:`input-components` documents the inputs that are pulled
for a model run.
:ref:`input-demographics` describes the demographic and location
inputs that need to be set for a model.
:ref:`covariates` describes how covariates are pulled and transformed.
:ref:`mtother` describes how we calculate other cause mortality
from the mortality inputs and use them as a constraint.
:ref:`measurement-inputs` documents how each of those
inputs works together to create one large object that stores all of the
input data for a model run (including each of the input components).


.. toctree::
   :maxdepth: 1

   input-demographics
   input-components
   covariates
   omega-constraint
   measurement-inputs
