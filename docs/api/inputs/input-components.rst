.. _input-components:

Input Components
^^^^^^^^^^^^^^^^

These are all of the inputs that are pulled for a model run.
Some may not be pulled depending on the settings (for example, some models
don't have cause-specific mortality data).


Crosswalk Version
"""""""""""""""""

.. autoclass:: cascade_at.inputs.data.CrosswalkVersion
   :members:
   :undoc-members:
   :show-inheritance:


Cause-Specific Mortality Rate
"""""""""""""""""""""""""""""

.. autoclass:: cascade_at.inputs.csmr.CSMR
   :members:
   :undoc-members:
   :show-inheritance:


.. autofunction:: cascade_at.inputs.csmr.get_best_cod_correct


All-Cause Mortality Rate
""""""""""""""""""""""""

.. autoclass:: cascade_at.inputs.asdr.ASDR
   :members:
   :undoc-members:
   :show-inheritance:


Population
""""""""""

.. autoclass:: cascade_at.inputs.population.Population
   :members:
   :undoc-members:
   :show-inheritance:
