.. _mulcov-statistics:

Compute Covariate Multiplier Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mulcov Statistics Script
""""""""""""""""""""""""

*(Note: mulcov is a short name for "covariate multiplier")*

Once we've done a sample on a database to get posteriors,
we can compute statistics of the covariate multipliers.

This is useful because we often like to use covariate
multiplier statistics at one level of the cascade
as a prior for the covariate multiplier estimation
in another level of the cascade.

.. autofunction:: cascade_at.executor.mulcov_statistics.get_mulcovs

.. autofunction:: cascade_at.executor.mulcov_statistics.compute_statistics

.. autofunction:: cascade_at.executor.mulcov_statistics.mulcov_statistics

Mulcov Statistics Cascade Operation
"""""""""""""""""""""""""""""""""""

.. autoclass:: cascade_at.cascade.cascade_operations.MulcovStatistics
   :members:
   :undoc-members:
   :show-inheritance:
