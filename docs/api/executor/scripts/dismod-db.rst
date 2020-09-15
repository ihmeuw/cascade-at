.. _dismod-db:

Dismod Database Creation and Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When we want to fill a dismod database with some data for a model,
and then run some commands on it, this is the script that we use.

We fill and extract dismod databases using :ref:`dismod-fill-extract` classes
and functions. Then the databases are filled according to their settings
and the arguments passed to these scripts, like whether to override the prior
in the settings with a parent prior (this is called "posterior to prior") or
whether to add a covariate multiplier prior.

Dismod Database Script
""""""""""""""""""""""

.. autofunction:: cascade_at.executor.dismod_db.dismod_db

.. autofunction:: cascade_at.executor.dismod_db.save_predictions

.. autofunction:: cascade_at.executor.dismod_db.fill_database

.. autofunction:: cascade_at.executor.dismod_db.get_mulcov_priors

.. autofunction:: cascade_at.executor.dismod_db.get_prior


Dismod Database Cascade Operation
"""""""""""""""""""""""""""""""""

.. autoclass:: cascade_at.cascade.cascade_operations._DismodDB
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: cascade_at.cascade.cascade_operations.Fit
   :members:
   :undoc-members:
   :show-inheritance:
