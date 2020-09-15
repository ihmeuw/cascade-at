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

.. autofunction:: cascade_at.executor.predict.fill_avgint_with_priors_grid

.. autoclass:: cascade_at.executor.predict.Predict
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: cascade_at.executor.predict.predict_sample_sequence

.. autofunction:: cascade_at.executor.predict.predict_sample_pool

.. autofunction:: cascade_at.executor.predict.predict_sample


Predict Cascade Operation
"""""""""""""""""""""""""

.. autoclass:: cascade_at.cascade.cascade_operations.Predict
   :members:
   :undoc-members:
   :show-inheritance:
