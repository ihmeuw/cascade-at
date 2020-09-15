.. _prior-specification:

Priors
------

These are classes for the priors.

.. py:class:: cascade_at.model.priors._Prior

   All priors have these methods.

   .. function:: parameters()

      Returns a dictionary of all parameters for this prior,
      including the prior type as "density".

   .. function:: assign(parameter=value, parameter=value...)

      Creates a new Prior object with the same parameters
      as this Prior, except for the requested changes.


.. autoclass:: cascade_at.model.priors.Uniform
   :members:

.. autoclass:: cascade_at.model.priors.Constant
   :members:

.. autoclass:: cascade_at.model.priors.Gaussian
   :members:

.. autoclass:: cascade_at.model.priors.Laplace
   :members:

.. autoclass:: cascade_at.model.priors.StudentsT
   :members:

.. autoclass:: cascade_at.model.priors.LogGaussian
   :members:

.. autoclass:: cascade_at.model.priors.LogLaplace
   :members:

.. autoclass:: cascade_at.model.priors.LogStudentsT
   :members:
