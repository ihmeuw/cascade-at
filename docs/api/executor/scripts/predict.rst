.. _predict:

Make Predictions of Integrands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we've fit to a database and/or made
posterior samples, we can make predictions
using the fit or sampled variables on the
average integrand grid. This is how we make predictions
for age groups and times on the IHME grid.

Predict Script
""""""""""""""

.. automodule:: cascade_at.executor.predict
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: main


Predict Cascade Operation
"""""""""""""""""""""""""

.. autoclass:: cascade_at.cascade.cascade_operations.Predict
   :members:
   :undoc-members:
   :show-inheritance:
