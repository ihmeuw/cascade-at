.. _grid-alchemy:

Grid Alchemy
------------

In order to build two-level models with the
settings from EpiViz-AT but at different
parent locations, and extracting the correct
information from the measurement inputs,
we use a wrapper around all of the modeling
components, with a method called
``construct_two_level_model``.

This alchemy object is one of the three
things that is read in each time we grab
a :ref:`model-context` object.

.. autoclass:: cascade_at.model.grid_alchemy.Alchemy
   :members:
   :undoc-members:
   :show-inheritance:

