.. _context:

Application Context
===================

Each model run needs to have an object that determines the file structure,
connections to the IHME databases, etc.

This context can be modified for a local environment, but that's not currently
implemented in an intuitive or user-friendly way. **When we want to enable local runs
of an entire cascade, this configuration is what we need to do design work on.**

Configuration
-------------

There is an additional repository that stores application
information for the IHME configuration.

.. autofunction:: cascade_at.context.configuration.application_config

.. _model-context:

Context
-------
Based on the configuration above, and a model version ID from the epi
database, we define a context object that keeps track of database
connections and file structures.

.. autoclass:: cascade_at.context.model_context.Context
   :members:
   :undoc-members:
   :show-inheritance:

It also provides methods to read in
the three things that are always needed to construct models:

* :ref:`measurement-inputs`
* :ref:`grid-alchemy`
* :ref:`settings-configuration`

