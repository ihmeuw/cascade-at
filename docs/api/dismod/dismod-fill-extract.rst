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

The :py:class:`~cascade_at.dismod.api.dismod_filler.DismodFiller`
uses the following table creation functions internally.

Formatting Reference Tables
"""""""""""""""""""""""""""

The dismod database needs some standard reference
tables. These are made with the following functions.

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.reference_tables.construct_integrand_table

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.reference_tables.default_rate_table

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.reference_tables.construct_node_table

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.reference_tables.construct_covariate_table

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.reference_tables.construct_density_table

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

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.grid_tables.construct_subgroup_table

Helper Functions
^^^^^^^^^^^^^^^^

Posterior to Prior
""""""""""""""""""

When we do "posterior to prior" that means to take
the fit from a parent database and use the rate posteriors as the prior
for the child fits. This happens in
:py:class:`~cascade_at.dismod.api.dismod_filler.DismodFiller`
when it builds the two-level model
with :py:class:`~cascade_at.model.grid_alchemy.Alchemy`
because it replaces the default
priors with the ones passed in.

The posterior is passed down by predicting the parent model on the rate
grid for the children. To construct the rate grid, we use the following
function:

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior.get_prior_avgint_grid

And then to upload those priors from the rate grid to the IHME databases
since the IHME databases require standard GBD ages and times, we use
this function. This is just for visualization purposes:

.. autofunction:: cascade_at.dismod.api.fill_extract_helpers.posterior_to_prior.format_rate_grid_for_ihme

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
