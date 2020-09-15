.. _dismod-fill-extract:

Fill and Extract Helpers
------------------------

In order to fill data into the dismod databases
in a meaningful way for the cascade, we have two
classes that are subclasses
of :py:class:`~cascade_at.dismod.api.dismod_io.DismodIO`
and provide easy functionality for filling tables
based on a model version's settings.

Dismod Filler
^^^^^^^^^^^^^

.. autoclass:: cascade_at.dismod.api.dismod_filler.DismodFiller
   :members:
   :undoc-members:
   :show-inheritance:


Dismod Extractor
^^^^^^^^^^^^^^^^

.. autoclass:: cascade_at.dismod.api.dismod_extractor.DismodExtractor
   :members:
   :undoc-members:
   :show-inheritance:

Table Creation
^^^^^^^^^^^^^^

Formatting Dismod Data Tables
"""""""""""""""""""""""""""""

There are helper functions to create data files.
Broke them up into small functions to help with unit testing.

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.data_tables.prep_data_avgint

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.data_tables.construct_data_table

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.data_tables.construct_gbd_avgint_table

Formatting Grid Tables
""""""""""""""""""""""

There are helper functions to create grid tables
in the dismod database. These are things like WeightGrid
and SmoothGrid.

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.grid_tables.construct_model_tables

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.grid_tables.construct_weight_grid_tables

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.grid_tables.construct_subgroup_tables

Helper Functions
^^^^^^^^^^^^^^^^

Multithreading
""""""""""""""
When we want to do multithreading on a dismod
database, we can define some process
that works, for example, on only
a subset of a database's data or samples, etc.
In order to do this work, there is a base class
here that is subclassed in
:py:class:`~cascade_at.executor.sample`
and
:py:class:`~cascade_at.executor.predict.Predict`
since there are tasks that can be done in parallel
on one database.

.. autoclass:: cascade_at.dismod.api.multithreading._DismodThread

.. autofunction:: cascade_at.dismod.api.multithreading.dmdismod_in_parallel
