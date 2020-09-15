.. _model-module:

Modeling
--------

The model module provides tools to build a Dismod-AT model with variables,
constraints, priors, and in the grid structure that Dismod-AT requires.

The main model object is documented here :ref:`model-class`.
The model object has two levels maximum (parents and children).
To build that model object with "global" settings from an EpiViz-AT
model, we have a wrapper around the model object, described below
in :ref:`alchemy` that builds a two-level model at any parent location
ID in a model hierarchy.

.. toctree::
   :maxdepth: 1

   var
   age-time-grid
   dismod-groups
   priors
   covariate
   smooth-grid
   grid-alchemy
